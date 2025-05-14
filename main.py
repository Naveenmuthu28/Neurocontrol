# BCI Project with both Frontend & Backend

import sys
import os
import json
import shutil
import threading

# Data Collection Libraries
import serial
from serial.tools import list_ports
import socket
import csv
import time
import datetime  
from tqdm import tqdm

# Model Training Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy import signal

# Live Prediction Libraries
from collections import deque, Counter

# UI Libraries
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QTextEdit, QTabWidget, QComboBox, QMessageBox, QProgressBar, QLineEdit, QInputDialog
)
from PySide6.QtCore import Qt, QTimer, Slot, QMetaObject, Q_ARG, QThread, Signal



#----------------------------------------------------------------- Backend Code ---------------------------------------------------------------------------#

class BCIBackend:

#----------------------------------- Variables Initialization -----------------------------------#

    # Global Variables
    ESP_CONFIG_FILE = "esp8266_config.json"
    BAUD_RATE = 115200

    # Init Variables

    def __init__(self):
        self.is_predicting = False  # Initialize this to prevent AttributeError
        self.COM_PORT = None
        self.ser = None  # 🔹 Store serial connection

#----------------------------------- Variables Initialization -----------------------------------#

#--------------------------------------------------- All Helper Functions ---------------------------------------------------#

    # Helper Functions for Data Collection and Live Prediction
    def serial_port(self):
        """Detects available serial ports."""
        ports = [port.device for port in list_ports.comports()]
        return ports[0] if ports else None

    def get_available_ports(self):
        """Returns a list of available serial ports."""
        ports = [port.device for port in list_ports.comports()]
        return ports

    def close_serial_connection(self):
        """Closes the serial connection if it is open."""
        if self.ser and self.ser.is_open:
            print("Closing COM Port now...")
            self.ser.close()
            self.ser = None  # 🔹 Clear reference to COM port
            print("COM Port Closed Successfully!")
        else:
            print("COM Port was already closed.")


    # Helper Functions for Model Training and Live prediction
    def setup_filters(self, sampling_rate):
        b_notch, a_notch = signal.iirnotch(50.0 / (0.5 * sampling_rate), 30.0)
        b_bandpass, a_bandpass = signal.butter(
            4, 
            [0.5 / (0.5 * sampling_rate), 30.0 / (0.5 * sampling_rate)], 
            "band"
        )
        return b_notch, a_notch, b_bandpass, a_bandpass

    def process_eeg_data(self, data, b_notch, a_notch, b_bandpass, a_bandpass):
        data = signal.filtfilt(b_notch, a_notch, data)
        data = signal.filtfilt(b_bandpass, a_bandpass, data)
        return data

    def calculate_psd_features(self, segment, sampling_rate):
        f, psd_values = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
        bands = {"alpha": (8, 13), "beta": (14, 30), "theta": (4, 7), "delta": (0.5, 3)}
        features = {}
        for band, (low, high) in bands.items():
            idx = np.where((f >= low) & (f <= high))
            features[f"E_{band}"] = np.sum(psd_values[idx])
        features["alpha_beta_ratio"] = (
            features["E_alpha"] / features["E_beta"] if features["E_beta"] > 0 else 0
        )
        return features

    def calculate_additional_features(self, segment, sampling_rate):
        f, psd = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
        peak_frequency = f[np.argmax(psd)]
        spectral_centroid = np.sum(f * psd) / np.sum(psd)
        log_f = np.log(f[1:])
        log_psd = np.log(psd[1:])
        spectral_slope = np.polyfit(log_f, log_psd, 1)[0]
        return {
            "peak_frequency": peak_frequency,
            "spectral_centroid": spectral_centroid,
            "spectral_slope": spectral_slope,
        }
    
    # Helper Function to get path properly in EXE/ELF
    def get_base_path(self):
        """Returns the correct base path whether running as a script or executable."""
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)  # Use dist/Neurocontrol/
        return os.path.dirname(os.path.abspath(__file__))  # Running as .py script

    def get_users_folder(self):
        """Ensures the 'users' folder exists and returns its path."""
        base_path = self.get_base_path()
        users_folder = os.path.join(base_path, "users")
        
        # Ensure 'users' directory exists, create it if not
        os.makedirs(users_folder, exist_ok=True)  
        
        return users_folder
    
    # Helper Functions for Live Prediction
    def load_model_and_scaler(self, person_name):
        """Loads trained SVM model & scaler from local files."""
        person_folder = os.path.join(self.get_users_folder(), person_name)  # Corrected path
        model_path = os.path.join(person_folder, "model.pkl")
        scaler_path = os.path.join(person_folder, "scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Model or scaler file not found for {person_name}. Train the model first.")
            return None, None

        try:
            with open(model_path, "rb") as f:
                clf = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            return clf, scaler
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def load_esp_config(self):
        """Loads ESP8266 configuration."""
        base_path = self.get_base_path()
        esp_config_path = os.path.join(base_path, "esp8266_config.json")

        if os.path.exists(esp_config_path):
            try:
                with open(esp_config_path, "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_esp_config(self, ip, port):
        """Saves ESP8266 IP and Port to a file."""
        base_path = self.get_base_path()
        esp_config_path = os.path.join(base_path, "esp8266_config.json")

        config = {"ip": ip, "port": port}
        with open(esp_config_path, "w") as file:
            json.dump(config, file)
        return "ESP8266 configuration saved successfully."

    def send_command(self, command):
        """Sends ON/OFF command to ESP8266."""
        config = self.load_esp_config()
        if "ip" not in config or "port" not in config:
            return "ESP8266 configuration not found."

        esp_ip = config["ip"]
        esp_port = int(config["port"])

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(command.encode(), (esp_ip, esp_port))
                print(f"Sent command: {command} to {esp_ip}:{esp_port}")
        except Exception as e:
            print(f"Error sending command: {e}")

        
    # Helper Functions for users list update and delete in Frontend
    def list_users(self):
        """Returns a list of users who have collected EEG data."""
        users_folder = self.get_users_folder()  # Use get_users_folder() instead of get_base_path()

        if not os.path.exists(users_folder):
            return []

        return [user for user in os.listdir(users_folder) if os.path.isdir(os.path.join(users_folder, user))]

    
    def delete_user(self, person_name):
        """Deletes a user's data folder."""
        folder_path = os.path.join(self.get_users_folder(), person_name)  # Use get_users_folder()

        if os.path.exists(folder_path):
            print("Deleting folder:", folder_path)  # Debugging print
            shutil.rmtree(folder_path)
            return f"User '{person_name}' deleted successfully."
        else:
            return f"User '{person_name}' not found."


    def delete_all_users(self):
        """Deletes all stored users and their EEG data."""
        users_folder = self.get_users_folder()  # Use get_users_folder()

        if os.path.exists(users_folder):
            shutil.rmtree(users_folder)
            os.makedirs(users_folder)  # Ensure it's recreated
            return "All users deleted successfully."
        return "No users found to delete."

    
#--------------------------------------------------- All Helper Functions ---------------------------------------------------#

    
#--------------------------------------------------------- Backend Function for Data Collection ---------------------------------------------------------#

    def collect_data(self, person_name, stop_flag=lambda: True):
        """Collects EEG data and saves it in CSV format, supports stopping early."""
        if not self.COM_PORT:
            return "No serial port selected!"

        folder_path = os.path.join(self.get_users_folder(), person_name)  # Use get_users_folder()
        os.makedirs(folder_path, exist_ok=True)
        csv_file_path = os.path.join(folder_path, "signal.csv")


        try:
            self.ser = serial.Serial(self.COM_PORT, self.BAUD_RATE, timeout=1)
            print(f"Connected to {self.COM_PORT} at {self.BAUD_RATE} baud.")

            with open(csv_file_path, "a", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)

                start_time = time.time()
                while time.time() - start_time < 300 and stop_flag():  # 🔹 Stop if requested
                    if self.ser is None or not self.ser.is_open:
                        print("Serial port closed unexpectedly. Stopping collection.")
                        break

                    try:
                        data = self.ser.readline().decode("latin-1").strip()
                        if data.isdigit():
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 🔹 Keep milliseconds
                            csvwriter.writerow([timestamp, data])  # 🔹 Save formatted timestamp
                    except serial.SerialException as e:
                        print(f"SerialException: {e}")
                        break
                    except Exception as e:
                        print(f"Unknown Error: {e}")
                        break

            # 🔹 Ensure Serial Port Closes Properly
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = None

        except serial.SerialException as e:
            return f"Error collecting data: SerialException: {e}"
        except Exception as e:
            return f"Error collecting data: {e}"

        return f"Data collection completed for {person_name}"
    
#--------------------------------------------------------- Backend Function for Data Collection ---------------------------------------------------------#
    

#--------------------------------------------------------- Backend Function for Model Training ----------------------------------------------------------#

    def train_model(self, person_name):
        """Trains an SVM model on the collected EEG data with feature extraction and proper evaluation."""

        folder_path = os.path.join(self.get_users_folder(), person_name)  # Use get_users_folder()
        csv_file_path = os.path.join(folder_path, "signal.csv")

        if not os.path.exists(csv_file_path):
            return "No data available for training"

        try:
            # Load the dataset
            print(f"Loading data from {csv_file_path}...")
            df = pd.read_csv(csv_file_path)


            # Check for missing values and fill them with the mean
            if df.isnull().any().any():
                print("Data contains NaN values. Filling with mean...")
                df.fillna(df.mean(), inplace=True)

            # Split data dynamically
            num_samples = len(df)
            half = num_samples // 2
            df.loc[:half, 'label'] = 1  # First half is attentive
            df.loc[half:, 'label'] = 0  # Second half is relaxed

            # Drop timestamp column and rename
            df.drop(columns=df.columns[0], axis=1, inplace=True)
            df.columns = ['raw_eeg', 'label']
            data = df['raw_eeg']
            labels_old = df['label']

            # Signal filtering parameters
            sampling_rate = 512
            nyquist = 0.5 * sampling_rate
            notch_freq = 50.0
            lowcut, highcut = 0.5, 30.0

            # Notch filter
            notch_freq_normalized = notch_freq / nyquist
            b_notch, a_notch = signal.iirnotch(notch_freq_normalized, Q=30)

            # Bandpass filter
            lowcut_normalized = lowcut / nyquist
            highcut_normalized = highcut / nyquist
            b_bandpass, a_bandpass = signal.butter(4, [lowcut_normalized, highcut_normalized], btype='band')

            # Process EEG segments
            features = []
            labels = []
            for i in tqdm(range(0, len(data) - 512, 256), desc="Processing EEG Segments", unit="segment"):
                segment = data[i:i+512].to_numpy()
                segment = signal.filtfilt(b_notch, a_notch, segment)
                segment = signal.filtfilt(b_bandpass, a_bandpass, segment)

                segment_features = self.calculate_psd_features(segment, sampling_rate)
                additional_features = self.calculate_additional_features(segment, sampling_rate)
                combined_features = {**segment_features, **additional_features}
                features.append(combined_features)
                labels.append(labels_old.iloc[i])

            # Convert features to DataFrame
            features_df = pd.DataFrame(features)
            features_df['label'] = labels

            # Scale data
            scaler = StandardScaler()
            X = scaler.fit_transform(features_df.drop('label', axis=1))
            y = features_df['label']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Ensure at least two classes exist
            if len(np.unique(y_train)) < 2:
                print("Insufficient classes for training. Ensure data contains both relaxed and attentive states.")
                return "Not enough data for training"

            # Train the SVM model with optimized parameters
            svc = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
            svc.fit(X_train, y_train)

            # Evaluate the model
            y_pred = svc.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            classification = classification_report(y_test, y_pred) 
            confusion = confusion_matrix(y_test, y_pred)

            # Save model and scaler
            model_path = os.path.join(folder_path, 'model.pkl')
            scaler_path = os.path.join(folder_path, 'scaler.pkl')

            with open(model_path, 'wb') as model_file:
                pickle.dump(svc, model_file)
            print(f"Model saved to {model_path}")

            with open(scaler_path, 'wb') as scaler_file:
                pickle.dump(scaler, scaler_file)
            print(f"Scaler saved to {scaler_path}")

            # Return formatted training results
            return (
                f"Model Training Complete for {person_name}\n\n"
                f"Test Accuracy: {test_accuracy:.2f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1 Score: {f1:.4f}\n"
                f"Classification Report:\n {classification}\n"
                f"Confusion Matrix:\n {confusion}\n"
            )

        except KeyboardInterrupt:
            return "Training interrupted"
        except Exception as e:
            return f"Error during training: {e}"

#--------------------------------------------------------- Backend Function for Model Training ----------------------------------------------------------#        


#--------------------------------------------------------- Backend Function for Live Prediction ---------------------------------------------------------#
   
    def predict(self, person_name, stop_flag):
        """Performs live EEG prediction with stop functionality."""
        print(f"Starting Live Prediction for {person_name}")

        # Load model and scaler
        clf, scaler = self.load_model_and_scaler(person_name)
        if clf is None or scaler is None:
            print("Model or scaler could not be loaded. Exiting function.")
            yield "Model Error"
            return

        if not self.COM_PORT:
            print("No serial port selected! Exiting prediction.")
            yield "No Serial Port"
            return

        # Initialize the serial connection
        try:
            ser = serial.Serial(self.COM_PORT, self.BAUD_RATE, timeout=1)
            print(f"Connected to {self.COM_PORT} for live data prediction.")
        except Exception as e:
            print(f"Failed to connect to {self.COM_PORT}: {e}")
            yield f"Serial Error: {e}"
            return

        sampling_rate = 512
        b_notch, a_notch, b_bandpass, a_bandpass = self.setup_filters(sampling_rate)
        buffer = deque(maxlen=512)
        prediction_buffer = deque(maxlen=10)

        try:
            while not stop_flag.is_set():  # Check stop condition
                try:
                    raw_data = ser.readline()

                    if raw_data:
                        raw_data = raw_data.decode('latin-1').strip()
                        try:
                            eeg_value = float(raw_data)
                            buffer.append(eeg_value)

                            if len(buffer) == 512:
                                buffer_array = np.array(buffer)
                                processed_data = self.process_eeg_data(buffer_array, b_notch, a_notch, b_bandpass, a_bandpass)

                                psd_features = self.calculate_psd_features(processed_data, sampling_rate)
                                additional_features = self.calculate_additional_features(processed_data, sampling_rate)
                                features = {**psd_features, **additional_features}

                                df = pd.DataFrame([features])
                                X_scaled = scaler.transform(df)
                                prediction = clf.predict(X_scaled)

                                prediction_buffer.append(prediction[0])

                                if len(prediction_buffer) == 10:
                                    most_occurred_prediction = int(float(Counter(prediction_buffer).most_common(1)[0][0]))
                                    print(f"Most Occurred Prediction from Last 10: {most_occurred_prediction}")

                                    if most_occurred_prediction == 0:
                                        self.send_command("OFF")
                                    elif most_occurred_prediction == 1:
                                        self.send_command("ON")

                                    yield str(most_occurred_prediction)  # Send result to frontend
                                    prediction_buffer.clear()

                                buffer.clear()

                        except ValueError:
                            continue

                    else:
                        print("No valid data received. Retrying...")
                        continue

                except Exception as e:
                    print(f'Error during prediction: {e}')
                    continue

        finally:
            ser.close()  # Close serial connection on exit
            print("🔌 Serial connection closed.")

        print("Live prediction stopped.")
        yield "Prediction Stopped"  # Ensure frontend receives the stop signal

#--------------------------------------------------------- Backend Function for Live Prediction ---------------------------------------------------------#

#----------------------------------------------------------------- Backend Code ---------------------------------------------------------------------------#        


        
#---------------------------------------------------------------- Frontend Code ---------------------------------------------------------------------------#

# ----------------------------------------------------------- Frontend Main Window -----------------------------------------------------------------#

class BCIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.backend = BCIBackend()
        self.setWindowTitle("Brain-Controlled Interface Dashboard")

        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        self.setGeometry(screen_size)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()
        self.tabs = QTabWidget()

        # Initializing all tabs
        self.home_tab = HomeTab(self)
        self.eeg_tab = DataCollectionTab(self.backend, self)
        self.model_tab = ModelTrainingTab(self.backend, self)
        self.predict_tab = LivePredictionTab(self.backend, self)
        self.delete_users_tab = DeleteUsersTab(self.backend, self)

        # Adding tabs to UI
        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.eeg_tab, "Data Collection")
        self.tabs.addTab(self.model_tab, "Model Training")
        self.tabs.addTab(self.predict_tab, "Live Prediction")
        self.tabs.addTab(self.delete_users_tab, "Delete Users")

        layout.addWidget(self.tabs)
        self.central_widget.setLayout(layout)

        # Ensure all UI updates correctly
        self.refresh_all_tabs()

    def refresh_all_tabs(self):
        """Refresh all dropdowns and UI elements dynamically."""
        self.eeg_tab.refresh_users_list()
        self.model_tab.refresh_users_list()
        self.predict_tab.refresh_users_list()
        self.delete_users_tab.refresh_users_list()

# ---------------------------------------------------------------- Main Window ---------------------------------------------------------------------#

#------------------------------------------------------------- Frontend for Home Tab ---------------------------------------------------------------#

# Home Tab UI
class HomeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel("Welcome to the Brain-Controlled Interface Dashboard")
        layout.addWidget(label)
        self.setLayout(layout)

#------------------------------------------------------------- Frontend for Home Tab ---------------------------------------------------------------#


#--------------------------------------------------------- Frontend for Data Collection ------------------------------------------------------------#

# Data Collection Thread
class DataCollectionWorker(QThread):
    data_collected = Signal(str)  # 🔹 Signal to send data collection results back to UI

    def __init__(self, backend, person_name):
        super().__init__()
        self.backend = backend
        self.person_name = person_name
        self.running = True  # 🔹 Control flag for stopping

    def run(self):
        """Runs EEG data collection in a separate thread using QThread."""
        result = self.backend.collect_data(self.person_name, stop_flag=lambda: self.running)
        self.data_collected.emit(result)  # 🔹 Send result back to UI

    def stop(self):
        """Stops EEG data collection."""
        print("Stopping Data Collection Worker...")
        self.running = False  # 🔹 Stop flag to exit the loop
        self.quit()
        self.wait()  # 🔹 Ensure thread cleanup
        print("Data Collection Worker Stopped!")


# Data Collection Tab UI
class DataCollectionTab(QWidget):
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.is_collecting = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Enter person’s name for data collection:")
        self.name_input = QLineEdit()
        self.start_button = QPushButton("Start Data Collection")
        self.stop_button = QPushButton("Stop")
        self.progress_bar = QProgressBar()
        self.back_button = QPushButton("Back")

        layout.addWidget(self.label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.back_button)
        self.setLayout(layout)

        # 🔹 Disable Stop button initially
        self.stop_button.setEnabled(False)

        # Button connections
        self.start_button.clicked.connect(self.start_data_collection)
        self.stop_button.clicked.connect(self.stop_data_collection)
        self.back_button.clicked.connect(self.go_back)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)

    def refresh_users_list(self):
        """Refreshes user lists in Model Training, Prediction, and Delete Users tabs."""
        parent = self.parentWidget().parent()
        if isinstance(parent, BCIApp):
            parent.model_tab.refresh_users_list()
            parent.predict_tab.refresh_users_list()
            parent.delete_users_tab.refresh_users_list()

    def start_data_collection(self):
        """Starts collecting EEG data using QThread to avoid timer errors."""
        self.person_name = self.name_input.text().strip()
        if not self.person_name:
            QMessageBox.warning(self, "Warning", "Please enter a valid name.")
            return

        # 🔹 Get available serial ports
        available_ports = self.backend.get_available_ports()
        if not available_ports:
            QMessageBox.critical(self, "Error", "No serial ports available! Check your device.")
            return

        selected_port, ok = QInputDialog.getItem(self, "Select Serial Port",
                                                "Available Serial Ports:", available_ports, 0, False)
        if not ok or not selected_port:
            QMessageBox.warning(self, "Warning", "Data collection canceled: No port selected.")
            return

        # 🔹 Set selected port in backend
        self.backend.COM_PORT = selected_port

        # 🔹 Reset and start the progress bar
        self.progress_bar.setValue(0)
        self.name_input.setDisabled(True)
        self.is_collecting = True

        # 🔹 Disable "Start" button & Enable "Stop" button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.timer.start(3000)  # 🔹 Updates every 3 seconds (300 sec = 100%)

        # 🔹 Start Data Collection in QThread (Instead of threading.Thread())
        self.collection_worker = DataCollectionWorker(self.backend, self.person_name)
        self.collection_worker.data_collected.connect(self.show_completion_message)
        self.collection_worker.start()

    def run_collection(self):
        """Runs data collection, ensures COM port closes if the user stops, and refreshes UI."""
        result = self.backend.collect_data(self.person_name)

        # 🔹 Ensure collection stops after 5 minutes
        self.is_collecting = False
        self.timer.stop()
        self.progress_bar.setValue(100)

        # 🔹 Re-enable input field
        self.name_input.setDisabled(False)

        # 🔹 Force UI update
        self.refresh_users_list()
        QApplication.processEvents()

        # 🔹 Refresh Model Training, Prediction, and Delete Users tabs
        parent = self.parentWidget().parent()
        while parent and not isinstance(parent, BCIApp):
            parent = parent.parentWidget()

        if isinstance(parent, BCIApp):
            print("Found BCIApp, refreshing tabs...")
            parent.model_tab.refresh_users_list()
            parent.predict_tab.refresh_users_list()
            parent.delete_users_tab.refresh_users_list()
            QApplication.processEvents()

            # 🔹 Force UI dropdown reset again after 200ms
            QTimer.singleShot(200, lambda: parent.model_tab.refresh_users_list())
            QTimer.singleShot(200, lambda: parent.predict_tab.refresh_users_list())
            QTimer.singleShot(200, lambda: parent.delete_users_tab.refresh_users_list())

        # 🔹 Show completion message safely in the main thread
        QMetaObject.invokeMethod(self, "show_completion_message", Qt.QueuedConnection, Q_ARG(str, result))

    @Slot(str)
    def show_completion_message(self, result):
        """Displays a message after data collection and refreshes user lists."""
        QMessageBox.information(self, "Data Collection", result)

        # 🔹 Reset progress bar to 0 after completion
        self.progress_bar.setValue(0)  

        # 🔹 Re-enable Name Input Box
        self.name_input.setDisabled(False) 

        # 🔹 Refresh user lists after collecting data
        self.refresh_users_list()
        QApplication.processEvents()  # Ensure UI updates immediately

        # 🔹 Ask if the user wants to collect more data
        reply = QMessageBox.question(self, "Continue?",
                                    "Do you want to collect more data?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            print("User chose to continue collecting data.")
            self.progress_bar.setValue(0)  # Reset progress bar
            self.start_data_collection()  # Restart data collection **on the same COM port**
            return  # 🔹 Exit function, so no "Stopped" message appears

        # 🔹 If user selects "No", close COM port and refresh UI
        print("User chose to stop collecting data.")

        # 🔹 Ensure serial connection is properly closed
        if hasattr(self.backend, "ser") and self.backend.ser:
            if self.backend.ser.is_open:
                print("Closing COM Port after data collection...")
                self.backend.ser.close()
                self.backend.ser = None  # 🔹 Clear reference to COM port
                print("COM Port Closed Successfully!")
            else:
                print("COM Port was already closed.")
        else:
            print("No active COM Port found.")

        # 🔹 Enable "Start" button after stopping
        self.start_button.setEnabled(True)  # 🔹 Now "Start" button is re-enabled!
        self.stop_button.setEnabled(False)  # 🔹 Keep "Stop" disabled

        # 🔹 Refresh UI & dropdowns
        parent = self.parentWidget().parent()
        while parent and not isinstance(parent, BCIApp):
            parent = parent.parentWidget()

        if isinstance(parent, BCIApp):
            print("Found BCIApp, refreshing tabs...")
            parent.model_tab.refresh_users_list()
            parent.predict_tab.refresh_users_list()
            parent.delete_users_tab.refresh_users_list()
            QApplication.processEvents()

        # 🔹 Show "Data Collection Stopped" message **ONLY if user selected No**
        QMessageBox.information(self, "Stopped", "Data collection stopped.")


    def stop_data_collection(self):
        """Stops EEG data collection early, closes COM port, and refreshes UI."""
        self.is_collecting = False
        self.timer.stop()
        self.progress_bar.setValue(0)  # 🔹 Reset progress bar to 0

        # 🔹 Re-enable Name Input Box
        self.name_input.setDisabled(False) 

        # 🔹 Disable Stop button, Enable Start button
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

        # 🔹 Ensure Data Collection Thread Stops Safely
        if hasattr(self, "collection_worker") and self.collection_worker.isRunning():
            print("Stopping Data Collection Thread...")
            self.collection_worker.stop()  # 🔹 Stop worker thread properly
            self.collection_worker.quit()
            self.collection_worker.wait()  # 🔹 Ensure thread cleanup
            print("Data Collection Thread Stopped Successfully!")

        # 🔹 Close Serial Connection Properly
        self.backend.close_serial_connection()

        # 🔹 Refresh UI & dropdowns
        parent = self.parentWidget()
        while parent and not isinstance(parent, BCIApp):
            parent = parent.parentWidget()

        if isinstance(parent, BCIApp):
            print("Found BCIApp, refreshing tabs...")
            parent.model_tab.refresh_users_list()
            parent.predict_tab.refresh_users_list()
            parent.delete_users_tab.refresh_users_list()
            QApplication.processEvents()

    def update_progress(self):
        """Updates the progress bar to match 5 minutes (300 seconds)."""
        elapsed_time = self.progress_bar.value() * 3  # Scale value (300 sec = 100%)
        
        if elapsed_time < 300 and self.is_collecting:
            self.progress_bar.setValue(self.progress_bar.value() + 1)
        else:
            self.timer.stop()

    def go_back(self):
        """Navigates back to the Home tab."""
        parent = self.parentWidget()
        while parent and not isinstance(parent, BCIApp):
            parent = parent.parentWidget()

        if isinstance(parent, BCIApp):
            print("Back Button Pressed - Switching to Home Tab")  # Debugging
            parent.tabs.setCurrentIndex(0)  # 🔹 Switch to Home Tab (Index 0)

#--------------------------------------------------------- Frontend for Data Collection ------------------------------------------------------------#


#--------------------------------------------------------- Frontend for Model Training -------------------------------------------------------------#

# Model Training Thread
class ModelTrainingWorker(QThread):
    progress_signal = Signal(int)
    result_signal = Signal(str)

    def __init__(self, backend, person_name):
        super().__init__()
        self.backend = backend
        self.person_name = person_name

    def run(self):
        """Runs model training and sends progress updates."""
        for i in range(1, 101, 10):  # Fake progress update
            self.progress_signal.emit(i)
            time.sleep(0.5)  # Simulate training time

        result = self.backend.train_model(self.person_name)  # Train model
        self.result_signal.emit(result)  # Send training result

# Model Training Tab UI
class ModelTrainingTab(QWidget):
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Select a user to train the model:")
        self.user_dropdown = QComboBox()
        self.refresh_users_list()

        self.train_button = QPushButton("Train Model")
        self.progress_bar = QProgressBar()
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)

        # 🔹 New Clear Button (Initially Disabled)
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.setEnabled(False)  # Disable at the start
        self.clear_button.clicked.connect(self.clear_results)  # Link button action

        self.back_button = QPushButton("Back")

        layout.addWidget(self.label)
        layout.addWidget(self.user_dropdown)
        layout.addWidget(self.train_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_box)
        layout.addWidget(self.clear_button)  # Add Clear Button ABOVE Back Button
        layout.addWidget(self.back_button)

        self.setLayout(layout)

        # Button connections
        self.train_button.clicked.connect(self.start_training)
        self.back_button.clicked.connect(self.go_back)

        # 🔹 Create a Timer for Progress Bar Updates
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.start_time = None  # Track training start time
        self.training_duration = None  # Store estimated training time

    def refresh_users_list(self):
        """Refreshes the dropdown list in Model Training Tab."""
        self.user_dropdown.clear()

        users = self.backend.list_users()

        if users:
            self.user_dropdown.addItems(users)
            self.user_dropdown.setCurrentIndex(-1)
        else:
            self.user_dropdown.addItem("No users found")

        self.user_dropdown.repaint()
        QApplication.processEvents()

    def start_training(self):
        """Starts training a model for the selected user."""
        selected_user = self.user_dropdown.currentText()
        if selected_user == "No users found":
            QMessageBox.warning(self, "Warning", "No data available for training.")
            return

        self.progress_bar.setValue(0)
        self.train_button.setDisabled(True)  # Disable Train button during training

        # Start Training Timer
        self.start_time = time.time()

        # Run training in a separate thread
        self.training_thread = threading.Thread(target=self.run_training, args=(selected_user,))
        self.training_thread.start()

        # Start Progress Bar Timer
        self.progress_timer.start(500)  # Updates every 500ms

    def run_training(self, person_name):
        """Runs model training in a separate thread and updates UI safely."""
        # Measure actual training time
        start_time = time.time()
        result = self.backend.train_model(person_name)
        self.training_duration = time.time() - start_time  # Get actual training duration

        # Ensure UI updates happen in the main thread
        QMetaObject.invokeMethod(self, "display_result", Qt.QueuedConnection, Q_ARG(str, result))

    def update_progress_bar(self):
        """Dynamically updates the progress bar based on actual training time."""
        if self.training_duration is None:
            return  # Don't update if training duration is unknown

        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.training_duration:
            self.progress_bar.setValue(100)  # Training is done, set to 100%
            self.progress_timer.stop()  # Stop updating progress
        else:
            progress_percentage = int((elapsed_time / self.training_duration) * 100)
            self.progress_bar.setValue(progress_percentage)

    @Slot(str)
    def display_result(self, result):
        """Handles completion of training, updates UI, and displays results in the UI."""
        self.result_box.clear()
        self.result_box.setPlainText(result)  # Show results in the UI text box

        # Enable Clear Button when results are displayed
        self.clear_button.setEnabled(True)

        QMessageBox.information(self, "Training Complete", "Model Training Completed Successfully!")

        # Ensure the progress bar reaches 100% smoothly
        self.update_progress_bar()
        self.progress_bar.setValue(100)

        # Reset progress bar after short delay
        QTimer.singleShot(1500, lambda: self.progress_bar.setValue(0))

        # Re-enable Train button
        self.train_button.setDisabled(False)

        # Reset training time tracking
        self.training_duration = None

    def clear_results(self):
        """Clears the training results and disables the clear button."""
        self.result_box.clear()
        self.clear_button.setEnabled(False)  # 🔹 Disable Clear Button when no results

    def go_back(self):
        """Navigates back to the Home tab."""
        parent = self.parentWidget()
        while parent and not isinstance(parent, BCIApp):
            parent = parent.parentWidget()

        if isinstance(parent, BCIApp):
            print("Back Button Pressed - Switching to Home Tab")
            parent.tabs.setCurrentIndex(0)  # Switch to Home Tab (Index 0)

#--------------------------------------------------------- Frontend for Model Training -------------------------------------------------------------#


#--------------------------------------------------------- Frontend for Live Prediction ------------------------------------------------------------#

# Live Prediction Thread
class LivePredictionWorker(QThread):
    prediction_result = Signal(str)

    def __init__(self, backend, person_name):
        super().__init__()
        self.backend = backend
        self.person_name = person_name
        self.stop_event = threading.Event()  # Thread-safe stop flag

    def run(self):
        """Runs live EEG prediction in a separate thread."""
        self.stop_event.clear()  # Ensure it's reset before starting
        for result in self.backend.predict(self.person_name, stop_flag=self.stop_event):  # Pass stop_event directly
            if self.stop_event.is_set():  # Break loop if stop is triggered
                break
            self.prediction_result.emit(result)

    def stop(self):
        """Stops live EEG prediction."""
        print("Stopping Live Prediction Worker...")
        self.stop_event.set()  # Set stop flag to signal backend to stop
        self.quit()
        self.wait()
        print("Live Prediction Worker Stopped!")

# Live Prediction Tab UI
class LivePredictionTab(QWidget):
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Select a person for live prediction:")
        self.user_dropdown = QComboBox()
        self.start_button = QPushButton("Start Live Prediction")
        self.stop_button = QPushButton("Stop Prediction")
        self.result_label = QLabel("Prediction: ")
        self.back_button = QPushButton("Back")

        layout.addWidget(self.label)
        layout.addWidget(self.user_dropdown)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.back_button)
        self.setLayout(layout)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_live_prediction)
        self.stop_button.clicked.connect(self.stop_live_prediction)
        self.back_button.clicked.connect(self.go_back)

        self.refresh_users_list()
        self.user_dropdown.currentIndexChanged.connect(self.user_selected)

    def refresh_users_list(self):
        """Refreshes the dropdown list in Model Training Tab."""
        self.user_dropdown.clear()
        users = self.backend.list_users()
        if users:
            self.user_dropdown.addItems(users)
        else:
            self.user_dropdown.addItem("No users found")
        self.user_dropdown.setCurrentIndex(-1)

    def user_selected(self):
        """Enables start button when a user is selected."""
        if self.user_dropdown.currentIndex() != -1:
            self.start_button.setEnabled(True)

    def start_live_prediction(self):
        """Handles live prediction workflow."""
        person_name = self.user_dropdown.currentText()
        if not person_name:
            QMessageBox.warning(self, "Warning", "No trained model selected.")
            return

        # Step 1: Select COM Port
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if not available_ports:
            QMessageBox.critical(self, "Error", "No serial ports available! Check your device.")
            return

        selected_port, ok = QInputDialog.getItem(self, "Select Serial Port",
                                                 "Available Serial Ports:", available_ports, 0, False)
        if not ok or not selected_port:
            QMessageBox.warning(self, "Warning", "Live prediction canceled: No port selected.")
            return
        self.backend.COM_PORT = selected_port

        # Step 2: ESP Config Handling
        esp_config_path = os.path.abspath("esp8266_config.json")
        esp_config = None

        if os.path.exists(esp_config_path):
            use_saved = QMessageBox.question(
                self, "ESP Config", "Use saved ESP config?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if use_saved == QMessageBox.StandardButton.Yes:  # Explicit check
                with open(esp_config_path, 'r') as f:
                    esp_config = json.load(f)

        if not esp_config:
            esp_config = self.ask_for_esp_config()
            if not esp_config:
                QMessageBox.warning(self, "Warning", "Invalid ESP Config.")
                return
            with open(esp_config_path, 'w') as f:
                json.dump(esp_config, f)

        # Step 3: Start Prediction
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.result_label.setText("Prediction: In Progress...")

        self.prediction_worker = LivePredictionWorker(self.backend, person_name)
        self.prediction_worker.prediction_result.connect(self.update_prediction)
        self.prediction_worker.start()

    def ask_for_esp_config(self):
        """Prompts user for ESP config (port & IP)."""
        port, ok1 = QInputDialog.getText(self, "ESP Port", "Enter ESP Port:")
        ip, ok2 = QInputDialog.getText(self, "ESP IP", "Enter ESP IP Address:")
        if ok1 and ok2 and port and ip:
            return {"port": port, "ip": ip}
        return None

    @Slot(str)
    def update_prediction(self, result):
        """Updates the UI with the latest prediction result."""
        self.result_label.setText(f"Prediction: {result}")

    def stop_live_prediction(self):
        """Stops live EEG prediction."""
        if hasattr(self, "prediction_worker") and self.prediction_worker.isRunning():
            self.prediction_worker.stop()
            self.prediction_worker.wait()  # Ensure it stops fully
            del self.prediction_worker
            print("Stopped Live Prediction")

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.result_label.setText("Prediction: Stopped")

    def go_back(self):
        """Navigates back to the Home tab."""
        parent = self.parentWidget()
        while parent and not isinstance(parent, BCIApp):
            parent = parent.parentWidget()
        if isinstance(parent, BCIApp):
            parent.tabs.setCurrentIndex(0)

#--------------------------------------------------------- Frontend for Live Prediction ------------------------------------------------------------#



#---------------------------------------------------------- Frontend for Delete Users --------------------------------------------------------------#

class DeleteUsersTab(QWidget):
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel("Select a user to delete:")
        self.user_dropdown = QComboBox()
        self.refresh_users_list()

        self.delete_button = QPushButton("Delete Selected User")
        self.delete_all_button = QPushButton("Delete All Users")
        self.back_button = QPushButton("Back")

        layout.addWidget(self.label)
        layout.addWidget(self.user_dropdown)
        layout.addWidget(self.delete_button)
        layout.addWidget(self.delete_all_button)
        layout.addWidget(self.back_button)
        self.setLayout(layout)

        self.delete_button.clicked.connect(self.delete_selected_user)
        self.delete_all_button.clicked.connect(self.delete_all_users)
        self.back_button.clicked.connect(self.go_back)

    def refresh_users_list(self):
        """Refreshes the dropdown list in Delete Users Tab."""
        self.user_dropdown.clear()

        users = self.backend.list_users()

        if users:
            self.user_dropdown.addItems(users)
            self.user_dropdown.setCurrentIndex(-1)  # Force UI update
        else:
            self.user_dropdown.addItem("No users found")

        self.user_dropdown.repaint()  # Force UI repaint
        QApplication.processEvents()  # Ensure immediate UI update

    def delete_selected_user(self):
        """Deletes the selected user and refreshes all tabs in real-time."""
        selected_user = self.user_dropdown.currentText()
        if selected_user == "No users found":
            QMessageBox.warning(self, "Warning", "No users available to delete.")
            return

        reply = QMessageBox.question(self, "Confirm Deletion",
                                    f"Are you sure you want to delete {selected_user}?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            result = self.backend.delete_user(selected_user)
            QMessageBox.information(self, "Deletion Status", result)

            print(f"Deleting user: {selected_user}")  # Debug print

            # 🔹 Force Refresh of Delete Users Dropdown
            self.refresh_users_list()
            self.user_dropdown.repaint()  # Manually trigger repaint

            # 🔹 Refresh Model Training & Live Prediction Tabs **Using Backend**
            parent = self.parentWidget().parent()
            while parent and not isinstance(parent, BCIApp):
                parent = parent.parentWidget()

            if isinstance(parent, BCIApp):
                print("Found BCIApp, refreshing all tabs...")
                parent.model_tab.refresh_users_list()  # Uses backend.list_users()
                parent.predict_tab.refresh_users_list()  # Uses backend.list_users()
                QApplication.processEvents()  # 🔹 Ensure UI refreshes instantly

    def delete_all_users(self):
        """Deletes all users and refreshes all tabs in real-time."""
        reply = QMessageBox.question(self, "Confirm Deletion",
                                    "Are you sure you want to delete all users?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            result = self.backend.delete_all_users()
            QMessageBox.information(self, "Deletion Status", result)

            # Force UI Update
            self.refresh_users_list()
            self.user_dropdown.repaint()  # Manually trigger repaint

            # Refresh all related tabs **Using Backend**
            parent = self.parentWidget().parent()
            while parent and not isinstance(parent, BCIApp):
                parent = parent.parentWidget()

            if isinstance(parent, BCIApp):
                print("Found BCIApp, refreshing all tabs...")
                parent.model_tab.refresh_users_list()  # Uses backend.list_users()
                parent.predict_tab.refresh_users_list()  # Uses backend.list_users()
                QApplication.processEvents()  # Ensure UI refreshes instantly

    def go_back(self):
        """Navigates back to the Home tab."""
        parent = self.parentWidget()
        while parent and not isinstance(parent, BCIApp):
            parent = parent.parentWidget()

        if isinstance(parent, BCIApp):
            print("Back Button Pressed - Switching to Home Tab")  # Debugging
            parent.tabs.setCurrentIndex(0)  # Switch to Home Tab (Index 0)

#---------------------------------------------------------- Frontend for Delete Users --------------------------------------------------------------#

#---------------------------------------------------------------- Frontend Code ---------------------------------------------------------------------------#




# Main Function
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BCIApp()
    window.show()
    sys.exit(app.exec())


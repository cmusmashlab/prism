from typing import Tuple, List
import numpy as np
from scipy.ndimage import label
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

import _pickle
import binascii
import base64
import glob
import tqdm

class SimpleThermalFeaturizer:
    """Extract simple universal thermal features for activity recognition"""
    
    def __init__(self):
        self.feature_names = []
        self.debug_features = {}  # Store features for visualization


    def process_b64_thermal_data(self, input_dir: str, output_dir: str, prefix: str = "flir"):
        """
        Process b64 thermal data
        """
        # get the thermal data
        thermal_files = sorted(glob.glob(f"{input_dir}/{prefix}_*.b64"))
        all_timestamps = []
        all_raw_data = []
        for thermal_file in thermal_files:
            filename_base = os.path.basename(thermal_file)
            file_data = []
            output_file = os.path.join(output_dir, f"{os.path.basename(thermal_file)}.processed.pb")
            if os.path.exists(output_file):
                processed_data = pickle.load(open(output_file, "rb"))
                for instance_ts, instance_data in tqdm.tqdm(processed_data, desc=f"Loading thermal ({filename_base}) from cache"):
                    all_timestamps.append(instance_ts)
                    all_raw_data.append(instance_data)
                continue
            
            remainder = ""
            num_chunks = 0
            pbar = tqdm.tqdm(desc=f"Processing thermal ({filename_base}) from raw", total=os.path.getsize(thermal_file))
            with open(thermal_file, "r") as myFile:
                while True:
                    chunk = [remainder]
                    chunk_found = False
                    while not chunk_found:
                        try:
                            line = myFile.readline()
                            if line == "":  # End of File
                                break
                        except:
                            break
                        if " ||" in line:
                            chunk_found = True
                            line, remainder = line.split(" ||")
                        chunk.append(line)
                    chunk = ''.join(chunk)
                    if chunk == "":
                        break
                    if len(chunk.split(" | ")) < 2:
                        print(f"Not enough chunk size, {len(chunk)}, {thermal_file}, already_processed {num_chunks} chunks")
                        break
                    ts, data = self.process_b64_chunk(chunk)
                    if ts == -1:
                        print(f"Error processing chunk: {chunk}, {thermal_file}")
                        break
                    all_timestamps.append(ts)
                    all_raw_data.append(data)
                    file_data.append((ts, data))
                    pbar.update(len(chunk))
            # save the file to pickle
            pickle.dump(file_data, open(output_file, "wb"))
            print(f"Processed {thermal_file} with {len(file_data)} chunks")
        return all_timestamps, all_raw_data

    def process_b64_chunk(self, chunk):
        """
        Process one line of data from thermal
        """
        ts, encoded_data = chunk.split(" | ")
        try:
            data = pickle.loads(base64.b64decode(encoded_data.encode()))
        except _pickle.UnpicklingError:
            # print(f"Found unpickling error in chunk, {chunk}")
            data = np.zeros((160, 120))
            ts = -1
        except binascii.Error:
            data = np.zeros((160, 120))
            ts = -1
        assert isinstance(data, np.ndarray)
        return (ts, data)
    
    def extract_features(self, window: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract simple universal features (20 total)"""
        # Validate input
        if len(window.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {window.shape}")
        
        # Preprocess window to handle missing values
        processed_window = self._preprocess_window(window)
        
        # Initialize feature dictionary
        features = {}
        
        # Extract all feature sets
        features.update(self.extract_signature_statistics(processed_window))      # 8 features
        features.update(self.extract_proximity_features(processed_window))        # 4 features  
        features.update(self.extract_global_dynamics(processed_window))           # 4 features
        features.update(self.extract_shape_motion_features(processed_window))     # 4 features
        
        # Convert to ordered array
        self.feature_names = list(features.keys())
        feature_array = np.array([features[name] for name in self.feature_names])
        
        return feature_array, self.feature_names
    
    def _preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """Preprocess thermal window to handle missing values"""
        processed_frames = []
        
        for frame in window:
            # Convert to float32
            processed_frame = frame.astype(np.float32)
            
            # Remove negative values and convert to nan
            processed_frame[processed_frame < 0] = np.nan
            
            # Fill nan values with inpainting if needed
            if np.any(np.isnan(processed_frame)):
                processed_frame = cv2.inpaint(processed_frame, np.isnan(processed_frame).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            
            processed_frames.append(processed_frame)
        
        return np.array(processed_frames)
    
    def extract_signature_statistics(self, window):
        """Extract basic statistics for each thermal signature"""
        features = {}
        
        # Define temperature ranges
        temp_ranges = {
            'human': (30, 40),
            'hot': (45, 150),
            'cold': (-5, 15),
            'ambient': (16, 29)
        }
        
        for sig_name, (min_temp, max_temp) in temp_ranges.items():
            sig_pixels_per_frame = []
            sig_max_temps = []
            
            for frame in window:
                # Find pixels in temperature range
                sig_mask = (frame >= min_temp) & (frame <= max_temp)
                sig_pixels_per_frame.append(np.sum(sig_mask))
                
                if np.any(sig_mask):
                    sig_max_temps.append(np.max(frame[sig_mask]))
                else:
                    sig_max_temps.append(min_temp)
            
            # Basic statistics
            features[f'{sig_name}_avg_area'] = np.mean(sig_pixels_per_frame) / frame.size  # Normalized area
            features[f'{sig_name}_max_intensity'] = np.max(sig_max_temps)
        
        return features
    
    def extract_proximity_features(self, window):
        """Extract human proximity to hot/cold regions"""
        features = {}
        
        human_hot_distances = []
        human_cold_distances = []
        
        for frame in window:
            # Get masks
            human_mask = (frame >= 30) & (frame <= 40)
            hot_mask = (frame >= 45) & (frame <= 150)
            cold_mask = (frame >= -5) & (frame <= 15)
            
            if np.any(human_mask):
                human_centroid = self.get_centroid(human_mask)
                
                # Distance to hot regions
                if np.any(hot_mask):
                    hot_centroid = self.get_centroid(hot_mask)
                    distance = np.linalg.norm(human_centroid - hot_centroid)
                    # Normalize by frame diagonal
                    normalized_distance = distance / np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                    human_hot_distances.append(normalized_distance)
                
                # Distance to cold regions  
                if np.any(cold_mask):
                    cold_centroid = self.get_centroid(cold_mask)
                    distance = np.linalg.norm(human_centroid - cold_centroid)
                    normalized_distance = distance / np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                    human_cold_distances.append(normalized_distance)
        
        # Aggregate proximity measures
        features['human_hot_min_distance'] = np.min(human_hot_distances) if human_hot_distances else 1.0
        features['human_cold_min_distance'] = np.min(human_cold_distances) if human_cold_distances else 1.0
        features['human_hot_contact'] = float(np.min(human_hot_distances) < 0.1 if human_hot_distances else False)
        features['human_cold_contact'] = float(np.min(human_cold_distances) < 0.1 if human_cold_distances else False)
        
        return features
    
    def extract_global_dynamics(self, window):
        """Extract global temperature change patterns"""
        features = {}
        
        # Global temperature statistics over time
        global_means = [np.mean(frame) for frame in window]
        global_maxes = [np.max(frame) for frame in window]
        global_ranges = [np.max(frame) - np.min(frame) for frame in window]
        
        # Temperature trends
        features['global_temp_trend'] = np.polyfit(range(len(global_means)), global_means, 1)[0]
        features['global_max_temp'] = np.max(global_maxes)
        
        # Temperature activity (how much variation)
        features['avg_temp_range'] = np.mean(global_ranges)
        
        # Immediate changes (for sudden events like opening fridge)
        if len(global_means) > 1:
            temp_changes = np.abs(np.diff(global_means))
            features['max_temp_change'] = np.max(temp_changes)
        else:
            features['max_temp_change'] = 0.0
        
        return features
    
    def extract_shape_motion_features(self, window):
        """Extract basic shape and motion characteristics"""
        features = {}
        
        # Human movement tracking
        human_centroids = []
        human_areas = []
        
        for frame in window:
            human_mask = (frame >= 30) & (frame <= 40)
            
            if np.any(human_mask):
                centroid = self.get_centroid(human_mask)
                human_centroids.append(centroid)
                human_areas.append(np.sum(human_mask))
            else:
                if human_centroids:  # Use last known position
                    human_centroids.append(human_centroids[-1])
                else:
                    human_centroids.append([0, 0])
                human_areas.append(0)
        
        # Human movement magnitude
        if len(human_centroids) > 1:
            movements = [np.linalg.norm(np.array(human_centroids[i+1]) - np.array(human_centroids[i])) 
                        for i in range(len(human_centroids)-1)]
            features['human_movement'] = np.sum(movements) / len(window)  # Average movement per frame
        else:
            features['human_movement'] = 0.0
        
        # Human size consistency (indicates stationary vs moving)
        if human_areas and max(human_areas) > 0:
            features['human_size_stability'] = 1.0 - np.std(human_areas) / np.mean(human_areas)
        else:
            features['human_size_stability'] = 0.0
        
        # Hot region characteristics
        hot_regions_count = []
        hot_compactness_scores = []
        
        for frame in window:
            hot_mask = (frame >= 45) & (frame <= 150)
            
            if np.any(hot_mask):
                # Count distinct hot regions using connected components
                _, num_regions = label(hot_mask)
                hot_regions_count.append(num_regions)
                
                # Simple compactness measure
                area = np.sum(hot_mask)
                bounding_box_area = (np.max(np.where(hot_mask)[0]) - np.min(np.where(hot_mask)[0]) + 1) * \
                                   (np.max(np.where(hot_mask)[1]) - np.min(np.where(hot_mask)[1]) + 1)
                compactness = area / bounding_box_area if bounding_box_area > 0 else 0
                hot_compactness_scores.append(compactness)
            else:
                hot_regions_count.append(0)
                hot_compactness_scores.append(0)
        
        features['avg_hot_regions'] = np.mean(hot_regions_count)
        features['hot_compactness'] = np.mean(hot_compactness_scores)
        
        return features
    
    def visualize_features(self, window: np.ndarray, features: dict, show_window=True):
        """Visualize extracted features for debug purposes"""
        if len(window) == 0:
            return None
        
        # Preprocess window first 
        processed_window = self._preprocess_window(window)
            
        # Create diagnostic info for header
        frame_count = len(processed_window)
        temp_min = np.nanmin([np.nanmin(frame) for frame in processed_window])
        temp_max = min(np.nanmax([np.nanmax(frame) for frame in processed_window]), 100.0)  # Cap at 100°C
        human_detected = any(features[k] > 0 for k in features.keys() if 'human' in k and 'avg_area' in k)
        hot_detected = any(features[k] > 0 for k in features.keys() if 'hot' in k and 'avg_area' in k)
        
        # Create figure with thermal frame and overlaid features (horizontal layout)
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f'Thermal Features | {frame_count} frames | {temp_min:.1f}°C-{temp_max:.1f}°C | Human:{"✓" if human_detected else "✗"} Hot:{"✓" if hot_detected else "✗"}', 
                     fontsize=14, fontweight='bold')
        
        # Left: Thermal frame with overlays (scaled to match height)
        ax1 = plt.subplot(1, 2, 1)
        self._plot_thermal_frame_with_overlays(ax1, processed_window[-1])
        
        # Right: Feature values overlaid horizontally
        ax2 = plt.subplot(1, 2, 2)
        self._plot_horizontal_features(ax2, features)
        
        plt.tight_layout()
        
        if show_window:
            # Convert matplotlib figure to OpenCV image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
            
            # Convert RGBA to BGR for OpenCV (drop alpha channel)
            img = cv2.cvtColor(buf[:,:,:3], cv2.COLOR_RGB2BGR)
            
            # Display using OpenCV
            cv2.imshow('Thermal Feature Visualization', img)
            cv2.waitKey(1)
            
        plt.close(fig)
        return fig
    
    def _plot_thermal_frame_with_overlays(self, ax, frame):
        """Plot thermal frame with temperature range overlays"""
        # Frame is already preprocessed, so no need to handle NaNs again
        img = frame.astype(np.float32)
        
        # Normalize to 0-255 range
        if img.max() > img.min():
            img_norm = 255 * (img - img.min()) / (img.max() - img.min())
        else:
            img_norm = np.zeros_like(img)
        
        # Apply color map
        img_colored = cv2.applyColorMap(img_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_colored)
        # Cap display temperature at 100°C
        temp_max_display = min(np.nanmax(frame), 100.0)
        ax.set_title(f'Thermal Frame\nTemp: {np.nanmin(frame):.1f}°C - {temp_max_display:.1f}°C')
        
        # Overlay temperature range masks
        temp_ranges = {
            'human': (30, 40, 'green'),
            'hot': (45, 150, 'red'), 
            'cold': (-5, 15, 'blue'),
            'ambient': (16, 29, 'yellow')
        }
        
        for sig_name, (min_temp, max_temp, color) in temp_ranges.items():
            mask = (frame >= min_temp) & (frame <= max_temp)
            if np.any(mask):
                # Get bounding box
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    # Draw bounding box
                    rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                           linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
                    
                    # Add centroid
                    centroid = self.get_centroid(mask)
                    ax.plot(centroid[1], centroid[0], 'o', color=color, markersize=6)
                    ax.text(centroid[1]+2, centroid[0]-2, sig_name, color=color, fontsize=8, weight='bold')
        
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        ax.axis('off')
    
    def _plot_horizontal_features(self, ax, features):
        """Plot feature values as horizontal layout with compact groups"""
        feature_groups = {
            'Signature': [k for k in features.keys() if any(sig in k for sig in ['human', 'hot', 'cold', 'ambient'])],
            'Proximity': [k for k in features.keys() if 'distance' in k or 'contact' in k],
            'Dynamics': [k for k in features.keys() if 'global' in k or 'temp_trend' in k or 'temp_change' in k or 'avg_temp_range' in k],
            'Motion': [k for k in features.keys() if 'movement' in k or 'stability' in k or 'regions' in k or 'compactness' in k]
        }
        
        # Create 4 horizontal sections for each group
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
        y_positions = [0.8, 0.6, 0.4, 0.2]  # 4 horizontal bands
        
        for i, (group_name, feature_list) in enumerate(feature_groups.items()):
            if not feature_list:
                continue
                
            y_pos = y_positions[i]
            
            # Group header
            ax.text(0.02, y_pos + 0.08, group_name, fontweight='bold', fontsize=10, 
                   transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
            
            # Feature values in horizontal list
            x_start = 0.02
            x_spacing = 0.23  # Space between features horizontally
            
            for j, feature_name in enumerate(feature_list):
                if j >= 4:  # Limit to 4 features per row to fit
                    break
                    
                value = features.get(feature_name, 0)
                
                # Create short display name
                short_name = feature_name.replace('_avg_area', '').replace('_max_intensity', '_max')\
                    .replace('human_', 'h_').replace('global_', 'g_').replace('_distance', '_dist')\
                    .replace('_contact', '_con').replace('_movement', '_mov').replace('_stability', '_stab')\
                    .replace('_compactness', '_comp').replace('_regions', '_reg')
                
                x_pos = x_start + (j * x_spacing)
                
                # Display feature name and value
                ax.text(x_pos, y_pos + 0.03, short_name, fontsize=8, fontweight='bold', 
                       transform=ax.transAxes)
                ax.text(x_pos, y_pos - 0.02, f'{value:.3f}', fontsize=8, 
                       color=colors[i], transform=ax.transAxes)
                
                # Small bar indicator
                bar_width = 0.15
                bar_height = 0.02
                # Normalize values for display
                if 'distance' in feature_name:
                    norm_value = min(value, 1.0)
                elif 'contact' in feature_name:
                    norm_value = value
                elif 'movement' in feature_name:
                    norm_value = min(value / 10.0, 1.0)
                else:
                    norm_value = min(abs(value) / 100.0, 1.0)
                
                # Draw small horizontal bar
                bar_x = x_pos
                bar_y = y_pos - 0.06
                ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * norm_value, bar_height, 
                                         facecolor=colors[i], alpha=0.7, transform=ax.transAxes))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Feature Values', fontsize=12, pad=10)
        ax.axis('off')  # Clean look without axes
    
    
    
    @staticmethod
    def get_centroid(mask):
        """Get centroid of a binary mask"""
        coords = np.where(mask)
        if len(coords[0]) > 0:
            return np.array([np.mean(coords[0]), np.mean(coords[1])])
        else:
            return np.array([0, 0])
        

    def generate_raw_windows(self, timestamps: list, thermal_data: list, windows: list, session_output_dir: str, selected_windows: list, thermal_rig):
        """
        Generate raw windows from thermal data
        """
        # create the session output dir
        os.makedirs(session_output_dir, exist_ok=True)

        # Ensure numpy arrays for easier indexing
        timestamps = np.array(timestamps)
        # thermal_data = np.array(thermal_data, dtype=object)

        window_timestamps = []
        feature_list = []
        for win_start, win_end in tqdm.tqdm(windows, desc="Extracting thermal features"):
            indices = np.where((timestamps >= win_start) & (timestamps < win_end))[0]
            if len(indices) == 0:
                continue
            else:
                if win_end in selected_windows:
                    instance_data = [(ts, data) for ts, data in zip(timestamps[indices], thermal_data[indices])]
                    window_instance_dir = f"{session_output_dir}/{win_end}"
                    os.makedirs(window_instance_dir, exist_ok=True)
                    # save the raw thermal windows as pickle file
                    window_pickle_path = f"{window_instance_dir}/{thermal_rig}.pkl"
                    with open(window_pickle_path, "wb") as f:
                        pickle.dump(instance_data, f)
                    # save the raw thermal windows as video
                    window_video_path = f"{window_instance_dir}/{thermal_rig}.mp4"
                    cv2_writer = cv2.VideoWriter(window_video_path, cv2.VideoWriter_fourcc(*'avc1'), 8, (160, 120))
                    for _, thermal_frame in instance_data:
                        # Convert to float32
                        img = thermal_frame.astype(np.float32)

                        # Remove negative values and convert to nan
                        img[img < 0] = np.nan

                        # Fill nan values with average of neighboring pixels
                        if np.any(np.isnan(img)):
                            img = cv2.inpaint(img, np.isnan(img).astype(np.uint8), 3, cv2.INPAINT_TELEA)

                        # Normalize to 0-255 range
                        if img.max() > img.min():  # Avoid division by zero
                            img = 255 * (img - img.min()) / (img.max() - img.min())
                        else:
                            img = np.zeros_like(img)

                        # Apply color map
                        img_colored = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)

                        cv2_writer.write(img_colored)
                    cv2_writer.release()
        return None
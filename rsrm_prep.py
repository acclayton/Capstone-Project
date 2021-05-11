#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# Import modules
import numpy as np
import mne
import os
from wget import download
from pathlib import Path
from pyprep.find_noisy_channels import NoisyChannels

# Dataset class that specifies file structure for different datasets
class Dataset:
    
    # Init method and get_file_format method need to be more general
    # Write a config file to ingest dataset specific information
    
    # Constructor assigns name, location, and size
    def __init__(self, name):
        '''Placeholder docstring (need to make more general)'''
        self.name = name
        
        # Data dictionary for all datasets: directory, subjects, tasks
        data_dict = {'motormovement_imagine': 
                         {'base_dir': 'https://github.com/VIXSoh/SRM/raw/master/data/',
                          'n_subj': 109,
                          'n_task': 14
                         }
                     # Add datasets here
                     # Write congig for different datasets
                    }
        # Assign attributes to dataset instance
        self.base_dir = data_dict[self.name]['base_dir']
        self.n_subj = data_dict[self.name]['n_subj']
        self.n_task = data_dict[self.name]['n_task']
    
    # Generates appropriate file paths
    def get_file_format(self, subj, task):
        '''Placeholder docstring (need to make more general)'''
        
        # Checks if name follows this file structure
        if self.name in ['motormovement_imagine']:
            
            # Get all file paths
            subj_num = f"S{str(subj).rjust(3, '0')}"
            task_num = f"R{str(task).rjust(2, '0')}"
            self.file_path = f'{self.base_dir}{subj_num}/{subj_num}{task_num}.edf'

    # Get data from internet using wget
    def wget_raw_edf(self):
        '''Reads an edf file from the internet (need to make more general for multiple file formats)

        Returns
        -------
        raw: mne raw object
            Raw dataset
        '''

        wg = download(self.file_path) # download the data locally (temporarily)
        print(wg)
        raw = mne.io.read_raw_edf(wg, preload=True) # read data as an MNE object
        os.remove(wg) # delete the file locally
        return raw

    # Specifies which files to iterate through
    def gen_iter(self, param, n_params):
        '''Helper method that allows for specific subject/task combinations or simply all of them

        Returns
        -------
        param_iter: list
            The iterator objects useful for subjects and tasks
        '''

        if param != 'all': # for subset of the data
            if type(param) != list:
                param_iter = [param]
            else:
                param_iter = param
        else: # for all of the data
            param_iter = range(1, n_params+1)
        return param_iter
    
    # Reads multiple EEG files with a single call
    def preproc(self, event_dict, baseline_start, stim_dur, montage, out_dir='preproc', subjects='all', tasks='all', 
                hp_cutoff=1, lp_cutoff=50, line_noise=60, seed=42, eog_chan='none'):
        '''Preprocesses a single EEG file. Assigns a list of epoched data to Dataset instance,
        where each entry in the list is a subject with concatenated task data. Here is the basic 
        structure of the preprocessing workflow:
        
            - Set the montage
            - Band-pass filter (high-pass filter by default)
            - Automatically detect bad channels
            - Notch filter out line-noise
            - Reference data to average of all EEG channels
            - Automated removal of eye-related artifacts using ICA
            - Spherical interpolation of detected bad channels
            - Extract events and epoch the data accordingly
            - Align the events based on type (still need to implement this!)
            - Create a list of epoched data, with subject as the element concatenated across tasks
        
        Parameters
        ----------
        event_dict: dict
            Maps integers to semantic labels for events within the experiment
            
        baseline_start: int or float
            Specify start of the baseline period (in seconds)
            
        stim_dur: int or float
            Stimulus duration (in seconds)
                Note: may need to make more general to allow various durations
                
        montage: mne.channels.montage.DigMontage
            Maps sensor locations to coordinates
            
        subjects: list or 'all'
            Specify which subjects to iterate through
            
        tasks: list or 'all'
            Specify which tasks to iterate through
            
        hp_cutoff: int or float
            The low frequency bound for the highpass filter in Hz
            
        line_noise: int or float
            The frequency of electrical noise to filter out in Hz
            
        seed: int
            Set the seed for replicable results
            
        eog_chan: str
            If there are no EOG channels present, select an EEG channel
            near the eyes for eye-related artifact detection
        '''

        missing = [] # initialize missing file list
        subj_iter = self.gen_iter(subjects, self.n_subj) # get subject iterator
        task_iter = self.gen_iter(tasks, self.n_task) # get task iterator

        # Iterate through subjects (initialize subject epoch list)
        epochs_subj = []
        for subj in subj_iter:

            # Iterate through tasks (initialize within-subject task epoch list)
            epochs_task = []
            for task in task_iter:
                # Specify the file format
                self.get_file_format(subj, task)

                try: # Handles missing files
                    raw = self.wget_raw_edf() # read
                except:
                    print(f'---\nThis file does not exist: {self.file_path}\n---')
                    # Need to write the missing file list out
                    missing.append(self.file_path)
                    break
                    
                # Standardize montage
                mne.datasets.eegbci.standardize(raw)
                # Set montage and strip channel names of "." characters
                raw.set_montage(montage)
                raw.rename_channels(lambda x: x.strip('.'))

                # Apply high-pass filter
                np.random.seed(seed)
                raw.filter(l_freq=hp_cutoff, h_freq=lp_cutoff, picks=['eeg', 'eog'])

                # Instantiate NoisyChannels object
                noise_chans = NoisyChannels(raw, do_detrend=False)

                # Detect bad channels through multiple methods
                noise_chans.find_bad_by_nan_flat()
                noise_chans.find_bad_by_deviation()
                noise_chans.find_bad_by_SNR()

                # Set the bad channels in the raw object
                raw.info['bads'] = noise_chans.get_bads()
                print(f'Bad channels detected: {noise_chans.get_bads()}')

                # Define the frequencies for the notch filter (60Hz and its harmonics)
                #notch_filt = np.arange(line_noise, raw.info['sfreq'] // 2, line_noise)

                # Apply notch filter
                #print(f'Apply notch filter at {line_noise} Hz and its harmonics')
                #raw.notch_filter(notch_filt, picks=['eeg', 'eog'], verbose=False)

                # Reference to the average of all the good channels 
                # Automatically excludes raw.info['bads']
                raw.set_eeg_reference(ref_channels='average')

                # Instantiate ICA object
                #ica = mne.preprocessing.ICA(max_iter=1000)
                # Run ICA
                #ica.fit(raw)

                # Find which ICs match the EOG pattern
                #if eog_chan == 'none':
                #    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
                #else:
                #    eog_indices, eog_scores = ica.find_bads_eog(raw, eog_chan, verbose=False)

                # Apply the IC blink removals (if any)
                #ica.apply(raw, exclude=eog_indices)
                #print(f'Removed IC index {eog_indices}')

                # Interpolate bad channels
                raw.interpolate_bads()
                
                # Specify pre-processing directory
                preproc_dir = Path(f'{out_dir}/ {subj}')

                # If directory does not exist, one will be created.
                if not os.path.isdir(preproc_dir):
                    os.makedirs(preproc_dir)

               # raw.save(Path(preproc_dir, f'subj{subj}_task{task}_raw.fif'), 
                #         overwrite=True)

                # Find events
                events = mne.events_from_annotations(raw)[0]

                # Epoch the data
                preproc_epoch = mne.Epochs(raw, events, tmin=baseline_start, tmax=stim_dur, 
                                   event_id=event_dict, event_repeated='error', 
                                   on_missing='ignore', preload=True)
                
                # Equalize event counts
                preproc_epoch.equalize_event_counts(event_dict.keys())
                
                # Rearrange and align the epochs
                align = [preproc_epoch[i] for i in event_dict.keys()]
                align_epoch = mne.concatenate_epochs(align)
                
                # Add to epoch list
                epochs_task.append(align_epoch)

            # Assuming some data exists for a subject
            # Concatenate epochs within subject
            concat_epoch = mne.concatenate_epochs(epochs_task)
            epochs_subj.append(concat_epoch)
        # Attaches a list with each entry corresponding to epochs for a subject
        self.epoch_list = epochs_subj
            
    def feature_engineer(self, step_size, freq_min, freq_max, num_freq_fams, time_max):
        
        '''Computes features and flattens epoch list into a matrix. Assings a list
        of matrices where each element is a subject and the matrix dimensionality is
        num_features x num_trials. Here is the feature engineering workflow:
        
            -Extract power using Morlet wavelets
        
        Parameters
        ----------
        step_size: int or float
            The size of the time-window within each epoch to compute features
            
        freq_min: int or float
            The minimum frequency to compute features for
            
        freq_max: int or float
            The maximum frequency to compute features for
                
        num_freq_fams: int
            The number of frequency families to compute
        '''

        # Define frequencies of interest (log-spaced)
        freqs = np.logspace(*np.log10([freq_min, freq_max]), num=num_freq_fams)
        # Get different number of cycles for each frequency
        n_cycles = freqs / 2.

        # Iterate through subjects
        subj_mats = []
        for idx, subj_data in enumerate(self.epoch_list):
            print(f'----- Feature engineering subject {idx} -----')
            power = mne.time_frequency.tfr_morlet(subj_data.crop(tmin=-.2, tmax=time_max), freqs=freqs, 
                               n_cycles=n_cycles, use_fft=True,
                               return_itc=False, n_jobs=1, average=False)


            # Iterate through time-windows
            for i in np.arange(0, 
                               #max(roi_data.times), 
                               max(subj_data.times),
                               step_size):
                # Create a copy of the power object to crop at time-window boundaries
                cropped = power.copy()
                # Crop the data for time-window of interest
                step_data = cropped.crop(tmin=i, tmax=i+step_size)

                # Retain shape after averaging (add 3rd axis back)
                step_avg = np.expand_dims(
                    # Average along the samples within the time-window
                    np.average(step_data.data, axis=3), 
                        axis=3)
                # Stack the time-windowed arrays along the 3rd axis
                if i == 0:
                    step_stack = step_avg
                else:
                    step_stack = np.concatenate((step_stack, step_avg), axis=3)

            shp = step_stack.shape
            # Reshape the data to align on events
            arr2mat = np.reshape(step_stack, (shp[0], shp[1]*shp[2]*shp[3]))
            subj_mats.append(arr2mat)
            print(f'Completed subject {idx}')
        self.feature_mats = subj_mats
# Robot Control Dataset

## Overview

The TD-NIRS and EEG data for this dataset was collected using a Kernel Flow headset.

Participants were asked to:
- Clench left fist
- Clench right fist
- Clench both fists
- Tap tongue
- Relax

## Data

The `data` folder contains numpy files. Each numpy file represents a 15s chunk of data.

### Timing

```
t=0             t=3                     t=15
rest period     stimulus presented      end of data
```

### Format

You can use the following to load a chunk:
```python
arr = np.load('/tmp/file.npz', allow_pickle=True)
```

There are 3 keys in this array:
```python
> list(arr.keys())

['feature_moments', 'feature_eeg', 'label']
```

### Labels

You can access the label with:
```python
> arr['label'].item()

{'label': 'Both Fists',
 'subject_id': 'fa7e4026',
 'session_id': 'bf56a42c',
 'duration': 9.411478996276855}
```

The labels are:
- `Right Fist`
- `Left First`
- `Both Firsts`
- `Tongue Tapping`
- `Relax`

The `subject_id` represents a unique participant. Chunks with the same `subject_id` came from the same participant.

The `session_id` represents a unique ID for the recording. Chunks with the same `session_id` came from the same recording.

The `duration` is the duration of the stimulus itself. The cue was presented at t=3 in the chunk and was removed `duration` seconds after. The participant was in a rest state for the rest of the chunk.

## EEG

You can access the EEG data with:
```python
> arr['feature_eeg'].shape

(7499, 6) # (num_samples, num_channels)
```

The first dimension has the samples. The EEG streams at 500Hz and 15 seconds at 500Hz is 7499 samples.

The second dimension corresponds to the 6 channels. The values are in microvolts (µV). Their locations are:
```
0     1     2     3     4    5
AFF6  AFp2  AFp1  AFF5  FCz  CPz
```

## TD-NIRS

You can access the TD-NIRS data with:
```python
> arr['feature_moments'].shape

(72, 40, 3, 2, 3) # (num_samples, num_modules, num_sds_ranges, num_wavelengths, num_moments)
```

The first dimension has the samples. The TD-NIRS streams at 4.76Hz and 15 seconds at 4.76Hz is 72 samples.

The second dimension corresponds to the 40 modules on the Kernel Flow headset. The moments data is averaged by module across channels where the module acted as a source. Their location on the head, when viewed from outside the headset and above, with the nose being at the top and the back of the head at the bottom, is:

<img alt="Kernel Flow Module Map" src="https://huggingface.co/datasets/KernelCo/robot_control/resolve/main/FlowModuleMap.png" width="500px" />

The third dimension corresponds to the 3 various SDSs (source-detector separations) used. The moments data is averaged across channels whose separation is within a range. The mapping to index is:
```
0: short channels from 0mm to 10mm
1: medium channels from 10mm to 25mm
2: long channels from 25mm to 60mm
```

The fourth dimension corresponds to the wavelengths in the Kernel Flow system. Each sample contains 2 wavelengths worth of data:
```
0: 690nm / red
1: 905nm / infrared
```

The fifth dimension corresponds to the 3 moments:
```
0: log10(sum) - logarithm of total intensity
1: mean time of flight - average arrival time of photons
2: variance/central moment - temporal broadening of the photon pulse
```
# Raw data

The raw inputs are ROOT files from NPS replay or Geant4+SIMC simulation. These
paths are available on the JLab filesystem and may not be visible outside the
JLab network.

## Locations

| Source | Format | JLab path |
| --- | --- | --- |
| HCANA replay | ROOT tree `T` | `/cache/hallc/c-nps/analysis/pass2/replays/updated/nps_hms_coin_{run}_{seg}_1_-1.root` |
| Geant4 + SIMC example | ROOT tree `nerd` | `/lustre24/expphy/volatile/eic/ckin/nps_geant4/nps_excl_pi0_x36_6_short.root` |

In the replay filename, `run` and `seg` are the run and segment numbers. The
`1` is retained for historical naming compatibility, and `-1` denotes a file
containing all requested events.

## HCANA replay ROOT structure

The replay converter reads tree `T` by default. Its important branches are:

| Branch | Structure and use |
| --- | --- |
| `Ndata.NPS.cal.fly.adcSampWaveform` | Number of valid entries in the flattened waveform buffer |
| `NPS.cal.fly.adcSampWaveform` | Repeated block records used as node features |
| `Ndata.NPS.cal.fly.block_clusterID` | Number of HCANA cluster-label entries |
| `NPS.cal.fly.block_clusterID` | Cluster ID indexed by block; `-1` means unassigned |
| `NPS.cal.vtpClusX/Y/E/Time/Size` | Recorded VTP cluster position, energy, time, and size, used by VTP matching |

### Flattened waveform encoding

`NPS.cal.fly.adcSampWaveform` is decoded as:

```text
block_id, num_samples, sample_0, ..., sample_(num_samples-1),
block_id, num_samples, sample_0, ...
```

The converter accepts unique block IDs from `0` through `1079` and expects
each retained waveform to have `110` samples. Invalid and duplicate block
records are skipped. A truncated record stops waveform parsing, while an event
with no waveform or a non-110-sample first waveform is rejected.

Blocks labelled `-1` in `NPS.cal.fly.block_clusterID` are omitted from the
cluster map used to construct targets and edges.

## Geant4 + SIMC ROOT structure

[SIMC](https://github.com/JeffersonLab/simc_gfortran) generates the Hall C
physics events. The
[HallC SIMC-Geant framework](https://github.com/avnishphy/HallC_SIMC_Geant)
transports those events through Geant4, simulates the NPS response,
reconstructs clusters, and writes the ROOT ntuple consumed here.

The simulation converter reads tree `nerd` by default. Its expected branches
fall into four groups:

| Group | Representative branches | Meaning |
| --- | --- | --- |
| Event and calorimeter | `evtNb`, `edep[1080]` | Event number and deposited energy for every NPS block |
| Photon truth | `phot1_*`, `phot2_*` | Hit flags, generation vertices, calorimeter impact points, cluster sizes/positions, and deposited energies |
| Reconstructed clusters | `nClusters`, `clust_E/X/Y/Size`, `clust_Signals` | Cluster summaries and block-level pulse data |
| SIMC kinematics | `Q2`, `W`, `t`, `Weight`, spectrometer variables, photon four-vectors, and vertex variables | Original generator-level event information |

The `clust_Signals` branch is a flat vector encoding a nested cluster
structure:

```text
cluster_id, number_of_blocks,
  block_id, number_of_pulses,
    pulse_time, pulse_energy,
    pulse_time, pulse_energy, ...
  block_id, number_of_pulses, ...
cluster_id, number_of_blocks, ...
```

Although the raw branch stores each pulse as `(time, energy)`, the simulation
converter arranges each output feature pair as `(energy, time)`.

Next: [Training data](./training.md) describes the graphs derived from these
ROOT sources.

# Data sources

| Resource | Source and role |
| --- | --- |
| Native duckweed reporters | Watanabe et al. 2023, DOI 10.1093/plphys/kiad218; 777 measured inputs and 506 qualified foreground references. Optical foreground/background components provide reference training. |
| Single- and dual-reporter transfer | Watanabe et al. 2021, DOI 10.1093/pcp/pcab037. The single-reporter cohort uses Green-pass inputs; dual-reporter inputs use their complete recorded windows. |
| SCN sources | Shan et al. 2020, DOI 10.1016/j.neuron.2020.07.012; https://github.com/YongliShan/Shan_Neuron_2020 at commit 59a08d7ebf8e069be28b18d61814c792eb037a55. No explicit data redistribution license is stated upstream. Source/mixture waveforms are downloaded and reconstructed locally using experiments/prepare_shan_mixtures.py --download-source; the source manifest records 120 content checksums. |
| Coupled clocks | Wang et al. 2026, DOI 10.1038/s41467-025-68218-x; model source v1.1.0, DOI 10.5281/zenodo.17861530. References are realized finite-window phase periods. |
| Fetal ECG | Abdominal and Direct Fetal ECG Database 1.0.0, https://physionet.org/content/adfecgdb/1.0.0/, DOI 10.13026/C2RP4B, ODC Attribution 1.0. Cite Jezewski et al. 2012, DOI 10.1515/bmt-2011-0130, and Pollard et al. 2026, DOI 10.1038/s44360-026-00096-z. |

Prepared observations, qualification states and source mappings are provided in the numerical data release, except SCN waveform-bearing inputs, which are obtained from the cited upstream repository and reconstructed locally. SCN comparison results and source mappings remain distributed. The data archive includes the updated preparation entry and software/REPRODUCING.md; extract it before running the preparation command. Original upstream archives are available at the cited sources. Original software uses MIT; third-party software and source data retain their own licenses and attribution.

## Citation

Plain text:
```
K. Sonmezoz and M. F. Amasyali, "Same Sentence Prediction: A new Pre-training Task for BERT," 2021 Innovations in Intelligent Systems and Applications Conference (ASYU), 2021, pp. 1-6, doi: 10.1109/ASYU52992.2021.9598954.
```


BibTex:
```
@INPROCEEDINGS{9598954,
  author={Sonmezoz, Kaan and Amasyali, Mehmet Fatih},
  booktitle={2021 Innovations in Intelligent Systems and Applications Conference (ASYU)}, 
  title={Same Sentence Prediction: A new Pre-training Task for BERT}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ASYU52992.2021.9598954}}
```

## Training

![image](https://user-images.githubusercontent.com/25352036/166552317-79ef0c25-4f09-4da3-80a7-c42f560aa731.png)


### Datasets

Corpus has been prepared by [the same way as in Berturk.](https://github.com/stefan-it/turkish-bert/issues/28)

- [Latest wikipedia dump](https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles.xml.bz2)
  - Latest wikipedia dump was 01 November 2021.
  - BERTurk used Wikipedia dump on 2 February 2020 for pre-training. 
- Kemal Oflazer's corpus
  - Private
  - Contact with [Mr Oflazer](https://www.andrew.cmu.edu/user/ko/) to get corpus.  
- [OSCAR](https://oscar-corpus.com/post/oscar-2019/)
  - Public
  - Deduplicated version has been used.
- [OPUS](https://opus.nlpl.eu)
   - Public
   - Various datasets:   
     - [Bible Uedin](https://opus.nlpl.eu/bible-uedin.php)
     - [GNOME](https://opus.nlpl.eu/GNOME.php)
     - JW300
     - [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles.php)
     - [OPUS All](https://opus.nlpl.eu/opus-100.php) (?)
        - Not sure whether this is the one that used in BERTurk pre-training
     - [QED](https://opus.nlpl.eu/QED.php) 
     - [SETIMES](https://opus.nlpl.eu/SETIMES.php)
     - [Tanzil](https://opus.nlpl.eu/Tanzil.php)
     - [Tatoeba](https://opus.nlpl.eu/Tatoeba.php)
     - [TED2013](https://opus.nlpl.eu/TED2013.php)
     - [Wikipedia](https://opus.nlpl.eu/Wikipedia.php)


### Changelog

- 03.05.2022:
    - Added citation 
- 03.11.2021: 
    - Added datasets used
    - Repo initialized

# Data

The images and metadata are not included in this repo (they are large and come from MIT's Place Pulse 2.0 project). Download them yourself and arrange them like this:

```
data/place-pulse-2.0/
├── studies.tsv       # study questions; the code uses the "safer" study
├── qscores.tsv       # TrueSkill scores: study_id, place_id, num_votes, trueskill.score
├── locations.tsv     # one row per photo: _id (location id), place_id
└── images/           # street-view photos, *.JPG
```

Image filenames must contain the location id as the third underscore-separated field (for example `lat_lng_<location_id>_....JPG`), which is how the code matches photos to rows in `locations.tsv`.

**Source:** MIT Media Lab, Place Pulse 2.0 (<http://pulse.media.mit.edu/>), introduced in Dubey et al., "Deep Learning the City," ECCV 2016. The dataset is also mirrored on Kaggle. Check the dataset's terms of use before redistributing any images.

You can put the data anywhere and point to it with `--data-path`.

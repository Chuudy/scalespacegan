#!/bin/bash

mkdir -p models

urls=(
    "https://scalespacegan.mpi-inf.mpg.de/files/models/moon_rec6.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/milkyway_rec6.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/spain_rec8.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/hima_rec8.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/rembrandt_rec8.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/moon_gen4.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/milkyway_gen4.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/spain_gen4.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/hima_gen4.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/rembrandt_gen4.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/sunflower_gen4.pkl"
    "https://scalespacegan.mpi-inf.mpg.de/files/models/bricks_gen4.pkl"
)

for url in "${urls[@]}"
do
    wget -P models "$url"
done

echo "Download complete."
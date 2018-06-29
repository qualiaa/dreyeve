#!/bin/sh

# resize garmin footage
for folder in DREYEVE_DATA/[0-9][0-9]; do
    for size in 112 448; do
        if [ ! -f "$folder"/garmin_resized_${size}.avi ]; then
            yes | ffmpeg -i "$folder"/video_garmin.avi -vf scale="${size}:${size}" "$folder"/garmin_resized_${size}.avi
        fi
    done
done

# resize mean frames
for folder in DREYEVE_DATA/[0-9][0-9]; do
    for size in 112 448; do
        if [ ! -f "$folder"/mean_frame_$size.png ]; then
            echo "Resizing mean frame: $folder" >&2
            convert "$folder"/mean_frame.png -resize "${size}x${size}!" "$folder"/mean_frame_$size.png
        fi
    done
done

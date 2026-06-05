# pluto-sdr

Pure-Python ADALM-Pluto SDR scripts (no GNU Radio).

## FDD raw image transfer (uncompressed, BPSK)

Run on two PCs, each with a Pluto. One PC sends an image, the other receives.
Image bytes travel on the data carrier (Pluto 1 -> Pluto 2); control/requests
travel on the control carrier (Pluto 2 -> Pluto 1).

```bash
# Pluto 1 (sender)
python pluto_image_fdd_raw.py --role tx --image photo.jpg

# Pluto 2 (receiver)
python pluto_image_fdd_raw.py --role rx --out-dir ./received
```

Auto-calibration runs on both radios at startup; start them within a few
seconds of each other. To skip calibration and set RF manually:

```bash
python pluto_image_fdd_raw.py --role tx --image photo.jpg --skip-cal --rx-gain 40 --tx-atten -20
```

Dependencies: `pip install pyadi-iio numpy scipy pylibiio Pillow`

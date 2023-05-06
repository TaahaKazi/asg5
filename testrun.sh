#!/usr/bin/env bash

pip3 install gym pyvirtualdisplay -q
sudo apt-get install -y xvfb python-opengl ffmpeg

pip3 install --upgrade setuptools --user -q
pip3 install ez_setup -q
pip3 install gym[atari] -q
pip3 install gym[accept-rom-license] -q

python taaha_9pm_1.py run1 > run1.log
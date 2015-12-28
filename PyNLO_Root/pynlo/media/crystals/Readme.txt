Status as of 6/3/2015

I've decided to structure crystal information as

crystals package
	- __init__.py loads up crystals
	- CrystalContainer contains skeletal CrystalInstance class
	- Crystal-specific files subclass CrystalInstance

So far I have implemented this for PPLN (PPLN.py) and AgGaSe. All other materials need to be adapted.
from instanseg import InstanSeg
instanseg_brightfield = InstanSeg("brightfield_nuclei", image_reader= "tiffslide", verbosity=1)

labeled_output = instanseg_brightfield.eval(image = "d8bfd1dafdc4.tif",
                                            save_output = True,
                                            save_overlay = True)
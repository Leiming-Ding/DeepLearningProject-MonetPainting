Use GANs to Create Monet Art

The purpose of this project is to use GANs to extract features from Monet art images and photos and use these features to generate picttures.

The dataset contains four kinds of files: monet_jpg (300 Monet paintings sized 256x256 in JPEG format); monet_tfrec (300 Monet paintings sized 256x256 in TFRecord format); 
photo_jpg (7038 photos sized 256x256 in JPEG format); photo_tfrec (7038 photos sized 256x256 in TFRecord format).

Exploratory data analysis

I first checked the number of images in each file. Then, I randomly selected some images to present. From the selected images and photos, we can see that there are many landscape paintings.

Model architecture

For building the GANs model, I first resized images to be 128x128 for training efficiency. Then, I create make_generator_model() to extract features from images.

Model architecture

For the parameter tuning, I mainly focused on changes of the epochs. When the epoch is set to 5, there are almost no obvious features. When the epoch is set 10, 
some colors appear. When the epoch is set to 20, more colors appear. They have some features from landcape paintings from Monet.

Takeaways and learning

It is important to set the image size. When the image size is 256x256, it can capture more information but the training will take longer.
It is important to pay attention to epochs. From 5, 10, 20, 50 to 150, I see more noticeable features. I think more epochs will be better. Still need to strike a balance between quality and efficiency
When dealing with image data, it is better to use GPU or TPU, much faster than CPU.

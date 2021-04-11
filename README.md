[![Build Status](https://travis-ci.org/klarEDA/klar-EDA.svg?branch=master)](https://travis-ci.org/klarEDA/klar-EDA)
# klar-eda

A python library for automated exploratory data analysis

## Overview

*Documentation* - https://klareda.github.io/klar-EDA/

*Presentation*  - https://youtu.be/FsDV6a-L-wo

The library aims to ease the data exploration and preprocessing steps and provide a smart and automated technique for exploratory analysis of the data

The library consists of the following modules
* CSV Data Visualization
* CSV Data Preprocessing
* Image Data Visualization
* Image Data Preprocessing

## Usage

You can install the test version of the library by the below command::

    $ pip3 install -i https://test.pypi.org/simple/ klar-eda    

The above mentioned modules can be used as below::

    >>> import klar_eda

### CSV Data Visualization

    >>> from klar_eda.visualization import visualize_csv
    
    >>> visualize_csv(<csv-file-path>) 
    
    OR
    
    >>> visualize_csv(<data-frame>)

### CSV Data Preprocessing

    >>> from klar_eda.preprocessing import preprocess_csv

    >>> preprocess_csv(<csv-file-path>) 
    
    OR
    
    >>> preprocess_csv(<data-frame>)

### Image Data Visualization

    >>> from klar_eda.visualization import visualize_images

    >>> ds = tfds.load('cifar10', split='train', as_supervised=True)
    >>> images = []
    >>> labels = []
    >>> for image, label in tfds.as_numpy(ds):
            h = randint(24, 56)
            w = randint(24, 56)
            image = cv2.resize(image, (w, h))
            images.append(image)
            labels.append(label)
    
    >>> visualize_images(images, labels)

### Image Data Preprocessing

    >>> from klar_eda.preprocessing import preprocess_images

    >>> preprocess_images(<images-folder-path>)

If you liked our project, it would be really helpful if you could share this project with others.

## Contributing

For contributing to this project, feel free to clone the repository::

    git clone https://github.com/klarEDA/klar-EDA.git

For installing the necessary packages, run the below command::

    $ pip3 install -r requirement.txt

### Documentation

To test the documentation in local::

    $ cd docsource/
    $ make html

To push the latest documentation in github::
    
    $ cd docsource/
    $ make github

## License

klar-eda is released under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).


Please feel free to contact us for any issues OR 
for discussion of future scope of the library at contact.klareda@gmail.com

## References

https://test.pypi.org/project/klar-eda/

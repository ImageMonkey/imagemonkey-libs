The following document describes how to easily upload images to ImageMonkey and tag them with arbitrary labels in a scripted manner. 

# Prerequisites

* Python 3 (should also work with Python 2.7, but I haven't tested it)
* requests needs to be installed (`pip install requests`)
* Pillow needs to be installed (`pip install Pillow`)

# Usage

* Create an ImageMonkey account here: https://imagemonkey.io/signup
* Login via https://imagemonkey.io/login
* Go to your Profile page 
![alt text](https://raw.githubusercontent.com/bbernhard/imagemonkey-libs/master/python/doc/snippets/img/profile.png)
* Create a new API token
![alt text](https://raw.githubusercontent.com/bbernhard/imagemonkey-libs/master/python/doc/snippets/img/add_token.png)
* Copy API token to clipboard
![alt text](https://raw.githubusercontent.com/bbernhard/imagemonkey-libs/master/python/doc/snippets/img/copy_token_to_clipboard.png)
* Download [donate_and_label.py](https://github.com/bbernhard/imagemonkey-libs/blob/master/python/snippets/donate_and_label.py) and [secrets.template](https://github.com/bbernhard/imagemonkey-libs/blob/master/python/snippets/secrets.template) and put them into the appropriate folder. **The Python scripts need to be on the same hierarchy level as your image folders.**

![alt text](https://raw.githubusercontent.com/bbernhard/imagemonkey-libs/master/python/doc/snippets/img/folder_structure.png)

* Rename `secrets.template` to `secrets.py` and insert your API token
* Open `donate_and_label.py` and change the global variable `FOLDERNAME` to the name of the folder you want to push. The script allows to push only one folder at the time. 
* Create a json file which contains a list of all labels that apply to the files in the folder. The json file needs to be at the same level as the folder and needs to have the same name as the folder (just with a `.json` suffix). 
The json file needs to have the following syntax: 
```
"labels" : [
    {
        "label": "first label",
        "annotatable": true
    },
    {
        "label": "second label",
        "annotatable": false
    }
]
```

The `annotatable` property allows you to specify whether you want to make the label annotatable. 

e.q: Assume we want to push all the images in the folder `apple` to ImageMonkey. As we have taken all the photos in the folder ourselves, we know that each photo shows an apple which is lying on a table with a knife nearby. 
So our json file could look like this:

```
"labels": [
    {
        "label": "apple",
        "annotatable": true
    },
    {
        "label": "table",
        "annotatable": true
    },
    {
        "label": "knife",
        "annotatable": true
    }
]
```

*Tip: It's a good idea to put images in the same folder, that share the most labels together.*

* In order to run the script, `cd` to the directory your Python script resides and start it with `python donate_and_label.py`
* The script now gives you a quick summary what it will be doing. **Please verify that you have specified the correct folder and label names.**



![alt text](https://raw.githubusercontent.com/bbernhard/imagemonkey-libs/master/python/doc/snippets/img/confirm.png)

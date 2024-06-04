
## Single-View Foul Recognition
This repo contains all the code and instructions necessary to reproduce the results of the MobiusLabs submission to the 2024 Soccernet Multi-view Foul Recognition challenge. It is a fork from the original repo [here](https://github.com/SoccerNet/sn-mvfoul).

While the challenge contains multiple views per foul, the proposed model solely relies on view 0, which is the live feed of the main camera. 

## Steps to reproduce the submitted result

### Get the dataset
Follow the [link](https://pypi.org/project/SoccerNet/) to easily download the SoccerNet pip package.

If you want to download the data and annotations, you will need to fill a [NDA](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to get the password.

Then use the API to downlaod the data:

```
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="enter password")
```
To obtain the data in 720p which is used in the proposed method, add version = "720p" to the input arguments. 
Unzip each folder while maintaining the naming conventions. (Train, Valid, Test, Chall).

### Install the dependencies
Run the following lines to install all the dependencies:
```
conda create -n vars python=3.9

conda activate vars

Install Pytorch with CUDA : https://pytorch.org/get-started/locally/

pip install SoccerNet

pip install -r requirements.txt

pip install pyav

```

### Get the model weights
Download the model weights from [here](https://drive.google.com/drive/folders/1Q-ycV8-C-oLKx2fGudIv3bUdbOyCz9sw?usp=drive_link), and put the `model_weights` folder into the root directory of this repo.

### Run the evaluation
Go to the root of this repo and run:

```
python main.py --path /path/to/720p/videos/ --GPU 0 --model_name MULTDIMSTACKER_VIEW0 --temp_stride 1 --pooling multidim_stacking --pre_model multidim_stacker --fp16 --decode_height 726 --N_frames 15 --only_evaluation 2 --path_to_model_weights './model_weights/10_model.pth.tar' --use_tta --view_mode only_view0
```


## License
See the [License](LICENSE) file for details.


## Citation

Please cite our work if you use this code:
TODO: Add technical report


and make sure to cite the original work this is based on as well.

```
@InProceedings{Held2023VARS,
    author    = {Held, Jan and Cioppa, Anthony and Giancola, Silvio and Hamdi, Abdullah and Ghanem, Bernard and Van Droogenbroeck, Marc},
    title     = {{VARS}: Video Assistant Referee System for Automated Soccer Decision Making From Multiple Views},
    booktitle = cvsports,
    month     = Jun,
    year      = {2023},
	publisher = ieee,
	address = seattle,
    pages     = {5085-5096}
}
```

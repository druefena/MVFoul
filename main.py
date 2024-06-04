import os
import logging
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
import torch
from dataset import MultiViewDataset

import torch.nn as nn
import torchvision.transforms as transforms
from model import MVNetwork




import torchvision.transforms as transforms

def get_train_augmentations(aug_params):

    mean = aug_params['norm_mean']
    std = aug_params['norm_std']
    model_input_size = aug_params['model_input_size']
    
    if(aug_params['grayscale'] is False):
        trainAugmentations = transforms.Compose([
                                transforms.RandomAffine(degrees=(1.5), translate=(0.0, 0.04), scale=(1, 1), interpolation=transforms.InterpolationMode.BILINEAR),
                                transforms.RandomPerspective(distortion_scale=0.1, interpolation=transforms.InterpolationMode.BILINEAR, p=0.5),
                                transforms.CenterCrop(model_input_size),
                                transforms.ColorJitter(brightness=0.3, saturation=0.3, contrast=0.3),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.Normalize(mean=mean, std=std)
                                ])
    else:
        trainAugmentations = transforms.Compose([
                                transforms.RandomAffine(degrees=(1.5), translate=(0.0, 0.04), scale=(1, 1), interpolation=transforms.InterpolationMode.BILINEAR),
                                transforms.RandomPerspective(distortion_scale=0.1, interpolation=transforms.InterpolationMode.BILINEAR, p=0.5),
                                transforms.CenterCrop(model_input_size),
                                transforms.ColorJitter(brightness=0.3, saturation=0.3, contrast=0.3),
                                transforms.RandomHorizontalFlip(p=0.5),
                                ])
    
    return trainAugmentations

def get_val_augmentations(aug_params):

    mean = aug_params['norm_mean']
    std = aug_params['norm_std']
    model_input_size = aug_params['model_input_size']

    if(aug_params['grayscale'] is False):
        valAugmentations = transforms.Compose([
                                transforms.CenterCrop(model_input_size),
                                transforms.Normalize(mean=mean, std=std)
                                ])
    else:
        valAugmentations = transforms.Compose([
                                transforms.CenterCrop(model_input_size),
                                ])
    
    return valAugmentations



def checkArguments():

    # args.num_views
    if args.num_views > 5 or  args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.pooling_type
    if args.pooling_type != 'max' and args.pooling_type != 'mean' and args.pooling_type != 'attention' and args.pooling_type != 'multidim_stacking':
        print("Could not find your desired argument for --args.pooling_type:")
        print("Possible arguments are: max or mean")
        exit()


def main(*args):

    if args:
        args = args[0]
        LR = args.LR
        gamma = args.gamma
        step_size = args.step_size
        temp_stride = args.temp_stride
        center_frame = args.center_frame
        temp_jitter = args.temp_jitter_train
        number_of_frames = args.N_frames
        mv_dropout = args.mv_dropout
        fl_alpha = args.fl_alpha
        fl_gamma = args.fl_gamma
        use_fp16 = args.fp16
        freeze_backbone = args.freeze_backbone
        use_tta = args.use_tta

        weight_decay = args.weight_decay
        view_mode = args.view_mode
        squeeze_frames = args.squeeze_frames
        skip_inter_layer = args.skip_inter_layer
        ignore_os_redcard = args.ignore_os_redcard
        ignore_challenge_action = args.ignore_challenge_action
        
        model_name = args.model_name
        pre_model = args.pre_model
        num_views = args.num_views
        os_weight = args.os_weight
        batch_size = args.batch_size
        path = args.path
        pooling_type = args.pooling_type
        weighted_loss = args.weighted_loss
        max_num_worker = args.max_num_worker
        max_epochs = args.max_epochs
        continue_training = args.continue_training
        only_evaluation = args.only_evaluation
        path_to_model_weights = args.path_to_model_weights
        decode_height = args.decode_height
        sqrt_weights = args.sqrt_weights
        video_resolution = (720, 1280) if 'VARS_720p' in path else (224, 398)
    else:
        print("EXIT")
        exit()

    if(use_fp16):
        print('Training with FP16')
        from train_fp16 import trainer, evaluation
    else:
        print('Training with FP32')
        from train import trainer, evaluation

    if(use_tta):
        print('Using Test time augmentation (TTA)')


    

    # Logging information
    # numeric_level = getattr(logging, 'INFO'.upper(), None)
    # if not isinstance(numeric_level, int):
    #     raise ValueError('Invalid log level: %s' % 'INFO')

    os.makedirs(os.path.join("models", os.path.join(model_name, os.path.join(str(num_views), os.path.join(pre_model, os.path.join(str(LR),
                            "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + "_G" + str(gamma) + "_Step" + str(step_size)))))), exist_ok=True)

    best_model_path = os.path.join("models", os.path.join(model_name, os.path.join(str(num_views), os.path.join(pre_model, os.path.join(str(LR),
                            "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + "_G" + str(gamma) + "_Step" + str(step_size))))))

    log_path = os.path.join(best_model_path, "logging.log")

    # Clear existing handlers from the root logger    
    logging.basicConfig(
        level=logging.DEBUG,#numeric_level,
        format="%(message)s",
        #"%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        )

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    logger.info(f"All parsed arguments: {vars(args)}")

    if(squeeze_frames):
        decode_width = decode_height
        aspect_ratio = 1.0
    else:
        aspect_ratio = np.amax(video_resolution)/np.amin(video_resolution)
        decode_width = int(np.round(decode_height*aspect_ratio))
    if pre_model == "mvit_v2_s":
        #transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        aug_params = {'model_input_size': [224, 224],
                        'norm_mean': [0.45, 0.45, 0.45], 
                        'norm_std' : [0.225, 0.225, 0.225],
                        'grayscale': False,
        }
        

    elif(pre_model=='multidim_stacker'):
        aug_params = {'model_input_size': [720, 1280],
                      'grayscale': True,
                        'norm_mean': [None, None, None], 
                        'norm_std' : [None, None, None],
        }
    elif(pre_model=='multidim_stacker_color'):
        aug_params = {'model_input_size': [int(decode_height-6), int(np.round(decode_width-6*aspect_ratio))],
                      'grayscale': False,
                        'norm_mean': [0.4850, 0.4560, 0.4060], 
                        'norm_std' : [0.2290, 0.2240, 0.2250],
        }
    else:
        raise ValueError('Backbone {} not implemented'.format(pre_model))
    
    train_augment = get_train_augmentations(aug_params)
    val_augment = get_val_augmentations(aug_params)

    logger.info('Video native resolution ({},{})'.format(video_resolution[0], video_resolution[1]))
    logger.info('Decoding frames at ({},{})'.format(decode_height, decode_width))
    logger.info('Train augmentations: {}'.format(train_augment))
    logger.info('Val/Test augmentations: {}'.format(val_augment))

    
    if only_evaluation == 0:
        #path, N_frames, temp_stride, center_frame, temp_jitter, split, num_views, decode_width, decode_height, transform_model=None, ignore_clip0_flag=False):
        dataset_Test2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Test'], 
                                         num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, view_mode=view_mode, 
                                         read_grayscale_images=aug_params['grayscale'])
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 1:
        dataset_Chall = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Chall'], 
                                         num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, view_mode=view_mode,
                                         read_grayscale_images=aug_params['grayscale'])

        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 2:
        dataset_Valid2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Valid'], 
                                          num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                          view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])        
        dataset_Test2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Test'], 
                                         num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                         view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])
        dataset_Chall = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Chall'], 
                                         num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                         view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 3:
        # Create Train Validation and Test datasets
        #(self, path, N_frames, temp_stride, center_frame, temp_jitter, split, num_views, transform=None, transform_model=None)
        dataset_Train = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=temp_jitter, splits=['Train'],
            num_views = num_views, decode_width=decode_width, decode_height=decode_height, transform_model=train_augment, 
            view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])
        dataset_Valid2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Valid'], 
                                          num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                          view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])
        dataset_Test2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Test'], 
                                         num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                         view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=batch_size, shuffle=True,
            num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 4:
        # Create Train Validation and Test datasets
        #(self, path, N_frames, temp_stride, center_frame, temp_jitter, split, num_views, transform=None, transform_model=None)
        dataset_Train = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, 
                                              temp_jitter=temp_jitter, splits=['Train', 'Valid'],
            num_views = num_views, decode_width=decode_width, decode_height=decode_height, transform_model=train_augment, 
            view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])
        dataset_Valid2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Valid'], 
                                          num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                          view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])
        dataset_Test2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Test'], 
                                         num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                         view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=batch_size, shuffle=True,
            num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    
    elif only_evaluation == 5:
        # Create Train Validation and Test datasets
        #(self, path, N_frames, temp_stride, center_frame, temp_jitter, split, num_views, transform=None, transform_model=None)
        dataset_Train = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, 
                                              temp_jitter=temp_jitter, splits=['Train', 'Valid', 'Test'],
            num_views = num_views, decode_width=decode_width, decode_height=decode_height, transform_model=train_augment, 
            view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])
        dataset_Valid2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Valid'], 
                                          num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                          view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])
        dataset_Test2 = MultiViewDataset(path=path, N_frames=number_of_frames, temp_stride=temp_stride, center_frame=center_frame, temp_jitter=0, splits=['Test'], 
                                         num_views = 5, decode_width=decode_width, decode_height=decode_height, transform_model=val_augment, 
                                         view_mode=view_mode, read_grayscale_images=aug_params['grayscale'])

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=batch_size, shuffle=True,
            num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    

    ###################################
    #       LOADING THE MODEL         #
    ###################################
    model = MVNetwork(net_name=pre_model, agr_type=pooling_type, mv_dropout=mv_dropout, freeze_backbone=freeze_backbone, 
                      skip_inter_layer=skip_inter_layer, num_frames=number_of_frames).cuda()

    if path_to_model_weights != "":
        path_model = os.path.join(path_to_model_weights)
        model_dict = torch.load(path_model)['state_dict']

        if(pre_model=='mvit_v2_s'):
            try:
                model.load_state_dict(model_dict)
            except:
                print('Cannot load all weights. Trying to load only backbone weights.')
                # Create a new dictionary containing only the backbone weights
                backbone_state_dict = {key.replace('mvnetwork.aggregation_model.', ''): value for key, value in model_dict.items() if key.startswith('mvnetwork.aggregation_model.')}
                # Load only the backbone weights into the model
                model.mvnetwork.aggregation_model.load_state_dict(backbone_state_dict)

        else:
            model.load_state_dict(model_dict)
        #print(res)

    if only_evaluation >=3:

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, 
                                    betas=(0.9, 0.999), eps=1e-07, 
                                    weight_decay=weight_decay, amsgrad=False)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=10E-7) #10E-6

        epoch_start = 0

        if continue_training:
            path_model = os.path.join(log_path, 'model.pth.tar')
            load = torch.load(path_model)
            model.load_state_dict(load['state_dict'])
            optimizer.load_state_dict(load['optimizer'])
            scheduler.load_state_dict(load['scheduler'])
            epoch_start = load['epoch']


        offence_severity_weights = dataset_Train.getWeights()[0]
        if(ignore_os_redcard):
            offence_severity_weights[-1] = 0

        action_weights = dataset_Train.getWeights()[1]
        if(ignore_challenge_action):
            action_weights[6] = 0

        if(sqrt_weights is True):
            offence_severity_weights = torch.FloatTensor([np.sqrt(x) for x in offence_severity_weights])
            action_weights = torch.FloatTensor([np.sqrt(x) for x in action_weights])
        

        if weighted_loss == 'Yes':
            
            criterion_offence_severity = nn.CrossEntropyLoss(weight=offence_severity_weights.cuda())
            criterion_action = nn.CrossEntropyLoss(weight=action_weights.cuda())
            
        else:
            criterion_offence_severity = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
        
        
        criterion = [criterion_offence_severity, criterion_action]


    # Start training or evaluation
    if only_evaluation == 0:
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
            use_tta=use_tta,
        ) 
        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    elif only_evaluation == 1:
        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
            use_tta=use_tta,
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)

    elif only_evaluation == 2:
        prediction_file = evaluation(
            val_loader2,
            model,
            set_name="valid",
            use_tta=use_tta,
        )

        results = evaluate(os.path.join(path, "Valid", "annotations.json"), prediction_file)
        print("VAL")
        print(results)



        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
            use_tta=use_tta,
        )

        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
            use_tta=use_tta,
        )

    else:
        trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion, 
                best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs, 
                use_tta=use_tta, os_weight=os_weight)
        
    return 0



if __name__ == '__main__':

    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--path',   required=True, type=str, help='Path to the dataset folder' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=20,     help='Maximum number of epochs' )
    parser.add_argument('--model_name',   required=False, type=str,   default="VARS",     help='named of the model to save' )
    parser.add_argument('--batch_size', required=False, type=int,   default=2,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--GPU',        required=False, type=int,   default=0,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=20, help='number of worker to load data')

    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--skip_inter_layer", required=False, action='store_true', help="Skip inter classification layer")
    parser.add_argument("--use_tta", required=False, action='store_true', help="Use test time augmentation (flip frames)")
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--pre_model", required=False, type=str, default="mvit_v2_s", help="Name of the pretrained model")
    parser.add_argument("--pooling_type", required=False, type=str, default="attention", help="Which type of pooling should be done")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the loss should be weighted")
    parser.add_argument("--fp16", required=False, action='store_true', help="Use fp16 training")
    parser.add_argument("--freeze_backbone", required=False, action='store_true', help="Freeze backbone weights")
    parser.add_argument("--view_mode", required=False, type=str, default='all_views', help="Which views to use") #all_views, ignore_view0, only_view0, only_view1
    parser.add_argument("--decode_height", required=False, type=int, default=256, help="Decode height")
    parser.add_argument("--squeeze_frames", required=False, action='store_true', help="Squeeze frames to decode_heightxdecode_height at decode")
    parser.add_argument("--sqrt_weights", required=False, action='store_true', help="Sqrt of the weights for weight balancing")
    parser.add_argument('--os_weight', required=False, type=float,   default=1.0, help='Weight of offense severity loss' )
    parser.add_argument("--ignore_os_redcard", required=False, action='store_true', help="Give zero weight to red card os class")
    parser.add_argument("--ignore_challenge_action", required=False, action='store_true', help="Give zero weight to challenge action class")

    parser.add_argument("--temp_stride", required=False, type=int, default=2, help="The temporal stride")
    parser.add_argument("--center_frame", required=False, type=int, default=75, help="The ID of the centerframe (def 75)") 
    parser.add_argument("--temp_jitter_train", required=False, type=int, default=4, help="Add some temporal jitter to the centerframe")
    parser.add_argument("--N_frames", required=False, type=int, default=16, help="Number of frames per view")
    parser.add_argument("--mv_dropout", required=False, type=float, default=0.3, help="Dropout on MVAR network")
    parser.add_argument("--fl_alpha", required=False, type=float, default=-1, help="Focal Loss Alpha")
    parser.add_argument("--fl_gamma", required=False, type=float, default=-1, help="Focal Loss Gamma")
    parser.add_argument("--step_size", required=False, type=int, default=5, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.5, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decacy")

    parser.add_argument("--only_evaluation", required=False, type=int, default=3, help="Only evaluation, 0 = on test set, 1 = on chall set, 2 = on both sets and 3 = train/valid/test")
    parser.add_argument("--path_to_model_weights", required=False, type=str, default="", help="Path to the model weights")

    args = parser.parse_args()

    print('Using GPU {}'.format(args.GPU))

    ## Checking if arguments are valid
    checkArguments()

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # Start the main training function
    start=time.time()
    main(args, False)

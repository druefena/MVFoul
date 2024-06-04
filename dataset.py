from torch.utils.data import Dataset
from random import random
import torch
import random

from data_loader_multiple import label2vectormerge_multiplesplits, clips2vectormerge_multiplesplits
#from torchvision.io.video import read_video
import decord
from decord import cpu
import torch
import random
import torchvision
from PIL import Image
import numpy as np
import cv2

#vid, _, _ = read_video(self.clips[index][index_view], output_format="THWC")


# def read_video_frames(video_path, N, stride, split, center=75, train_jitter=2):

#     vr = decord.VideoReader(video_path, ctx=cpu(0))
    
#     if(split=='Train'):
#         temp_jitter = random.randint(-train_jitter, train_jitter)
#     else:
#         temp_jitter = 0

#     start = center -N//2*stride +1 + temp_jitter
#     end = start + N*stride

#     frame_ids = list(range(start,end, stride))
#     #print(frame_ids)

#     frames = torch.from_numpy(vr.get_batch(frame_ids).asnumpy(), dtype='float32')

#     import pdb; pdb.set_trace()
    
#     return frames

# ERRONEOUS_CLIPS_AT_720P = [
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_359/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_338/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_1409/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_46/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_442/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_89/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_1520/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_818/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_591/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_213/clip_2.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_565/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_1106/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_508/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_1465/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_773/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Test/action_86/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Test/action_96/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Test/action_138/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Test/action_236/clip_0.mp4',
# '/nas/dominic/challenges/Soccernet2024/sn-mvfoul/VARS_720p/mvfouls/Train/action_89/clip_0.mp4',
# ]



# def read_video_frames(video_path, N, stride, split, center=75, train_jitter=2):

#     if(video_path in ERRONEOUS_CLIPS_AT_720P):
#         vr = decord.VideoReader(video_path.replace('VARS_720p', 'VARS_224p'), ctx=decord.cpu(0))
#     else:
#         vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        
    
#     if split == 'Train':
#         temp_jitter = random.randint(-train_jitter, train_jitter)
#     else:
#         temp_jitter = 0

#     start = center - N // 2 * stride + 1 + temp_jitter
#     end = start + N * stride

#     frame_ids = list(range(start, end, stride))
    
#     # Read frames as float32 tensors
#     frames = []
#     for frame_id in frame_ids:
#         frame_data = vr[frame_id].asnumpy()  # Read frame data as numpy array
#         frame_tensor = torch.from_numpy(frame_data.astype('uint8'))  # Convert to float32 tensor
#         frames.append(frame_tensor)

#     frames = torch.stack(frames)  # Stack frames along the first dimension
    

#     return frames

def read_video_frames(video_path, N, stride, splits, center=75, train_jitter=2, W=640, H=480, convert_to_grayscale=False):

    try:
        vr = decord.VideoReader(video_path, width=W, height=H)
    except:
        vr = decord.VideoReader(video_path.replace('VARS_720p', 'VARS_224p'), width=W, height=H)

    if(len(vr) < 100): #Sometimes the high-res video is not yet read properly, yet does not throw an error. No idea why, but this just reads the low res video in such caes.
        vr = decord.VideoReader(video_path.replace('VARS_720p', 'VARS_224p'), width=W, height=H)
         
    
    if 'Train' in splits:
        temp_jitter = random.randint(-train_jitter, train_jitter)
    else:
        temp_jitter = 0

    start = center - N // 2 * stride + 1 + temp_jitter
    end = start + N * stride

    frame_ids = list(range(start, end, stride))
    
    # Read frames as float32 tensors
    frames = []
    for frame_id in frame_ids:
        try:
            frame_data = vr[frame_id].asnumpy()  # Read frame data as numpy array
        except:
            print('ERROR WITH VIDEO {} len(vr)={}, start={}'.format(video_path, len(vr), start))
        if convert_to_grayscale:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2GRAY)  # Convert to grayscale [H,W]
            frame_data = frame_data[:, :, np.newaxis] #Add additional dimension to keep color axis [H,W,1]
        
        frame_tensor = torch.from_numpy(frame_data.astype(float))/255.0  # Convert to float32 tensor
        frames.append(frame_tensor)

    frames = torch.stack(frames)  # Stack frames along the first dimension
    

    return frames

class MultiViewDataset(Dataset):
    def __init__(self, path, N_frames, temp_stride, center_frame, temp_jitter, splits, num_views, decode_width, decode_height, 
                 transform_model=None, view_mode='all_views', read_grayscale_images=False):

        if splits[0] != 'Chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking, self.number_of_actions = label2vectormerge_multiplesplits(path, splits, num_views)
            self.clips = clips2vectormerge_multiplesplits(path, splits, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            print('Offence severity distribution: {}'.format(self.distribution_offence_severity))
            print('Action distribution: {}'.format(self.distribution_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
        else:
            self.clips = clips2vectormerge_multiplesplits(path, splits, num_views, [])

        # INFORMATION ABOUT SELF.LABELS_OFFENCE_SEVERITY
        # self.labels_offence_severity => Tensor of size of the dataset. 
        # each element of self.labels_offence_severity is another tensor of size 4 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if it is not an offence, then the tensor is [1, 0, 0, 0]. 

        # INFORMATION ABOUT SELF.LABELS_ACTION
        # self.labels_action => Tensor of size of the dataset. 
        # each element of self.labels_action is another tensor of size 8 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if the action is a tackling, then the tensor is [1, 0, 0, 0, 0, 0, 0, 0]. 

        # INFORMATION ABOUT SLEF.CLIPS
        # self.clips => list of the size of the dataset
        # each element of the list is another list of size of the number of views. The list contains the paths to all the views of that particular action.

        # The offence_severity groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_offence_severity tensor
        # The type of action groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_action tensor
        
        self.splits = splits

        self.N_frames = N_frames
        self.temporal_stride = temp_stride
        self.center_frame = center_frame
        self.temp_jitter = temp_jitter
        self.decode_width = decode_width
        self.decode_height = decode_height
        self.read_grayscale_images = read_grayscale_images
        self.view_mode = view_mode
        
        #self.start = start
        #self.end = end
        #self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views

        #self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.clips)
        print(self.length)

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action, 
    def getWeights(self):
        return self.weights_offence_severity, self.weights_action, 


    # RETURNS
    #
    # self.labels_offence_severity[index][0] => tensor of size 4. Example [1, 0, 0, 0] if the action is not an offence
    # self.labels_action[index][0] => tensor of size 8.           Example [1, 0, 0, 0, 0, 0, 0, 0] if the type of action is a tackling
    # videos => tensor of shape V, C, N, H, W with V = number of views, C = number of channels, N = the number of frames, H & W = height & width
    # self.number_of_actions[index] => the id of the action
    #
    def __getitem__(self, index):

        prev_views = []

        if(self.view_mode=='all_views'):
            view_ids = range(len(self.clips[index]))
        elif(self.view_mode=='ignore_view0'):
            view_ids = range(1, len(self.clips[index]))
        elif('only_view' in self.view_mode):
            
            view_number = int(self.view_mode[-1])
            assert(view_number in [0, 1])
            view_ids = [view_number]
        else:
            raise NotImplementedError('View mode {} does not exist'.format(self.view_mode))


        for count, num_view in enumerate(view_ids): #range(len(self.clips[index])):

            index_view = num_view

            if len(prev_views) == np.amin([len(view_ids), 2]):
                continue

            # As we use a batch size > 1 during training, we always randomly select two views even if we have more than two views.
            # As the batch size during validation and testing is 1, we can have 2, 3 or 4 views per action.
            cont = True
            if 'Train' in self.splits:
                while cont:
                    #aux = random.randint(0,len(self.clips[index])-1)
                    aux = random.choice(view_ids)
                    if aux not in prev_views:
                        cont = False
                index_view = aux
                prev_views.append(index_view)


            # video, _, _ = read_video(self.clips[index][index_view], output_format="THWC")
            # frames = video[self.start:self.end,:,:,:]

            # final_frames = None

            # for j in range(len(frames)):
            #     if j%self.factor<1:
            #         if final_frames == None:
            #             final_frames = frames[j,:,:,:].unsqueeze(0)
            #         else:
            #             final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)
            
            #print(self.clips[index][index_view])
            final_frames = read_video_frames(self.clips[index][index_view], self.N_frames, self.temporal_stride, self.splits, 
                                             self.center_frame, self.temp_jitter, W=self.decode_width, H=self.decode_height,
                                             convert_to_grayscale=self.read_grayscale_images) #N,H,W,C
            
            #print('init',  final_frames.shape)
            final_frames = final_frames.permute(0, 3, 1, 2) #N,C,H,W
            #final_frames = final_frames.transpose(0, 3, 1, 2) #N,C,H,W

            #print(final_frames.shape)

            # for i in range(len(final_frames)):
            #     # Convert tensor to PIL image
            #     pil_image = tensor_to_pil_image(final_frames[i])
            #     pil_image.save('./orig_{:03d}.png'.format(i))
                #torchvision.utils.save_image(final_frames[i]*255.0, './orig_{:03d}.png'.format(i))

            # print('Before transform')
            # for c in range(3):
            #     mean_c = np.mean(np.array(final_frames[:,c,:,:]))
            #     std_c = np.std(np.array(final_frames[:,c,:,:]))
            #     print(mean_c, std_c)

            # if self.transform is not None:
            #     final_frames = self.transform(final_frames)#N,C,H,W

            #import pdb; pdb.set_trace()

            final_frames = self.transform_model(final_frames) #N, C, 224, 224 #C,N,224,224

            
            
            #final_frames = final_frames.permute(1, 0, 2, 3) #N,C,224,224
            #final_frames = final_frames.float()/255.0
            
            
            if count == 0:
                videos = final_frames.unsqueeze(0) #1,N,C,224,224
            else:
                final_frames = final_frames.unsqueeze(0)
                videos = torch.cat((videos, final_frames), 0)

        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()   

        videos = videos.permute(0, 2, 1, 3, 4) #Views,C,N,224,224

        if self.splits[0] != 'Chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length
    

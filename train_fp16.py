import logging
import os
import time
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np

logger = logging.getLogger(__name__)

def prettyprint(res_dict):
    def custom_formatter(x):
        # Format each element with 4 characters before and 2 characters after the floating point dot
        return f"{x:6.2f}"
    
    # Iterate over the dictionary items and print each key-value pair
    for key, value in res_dict.items():
        # If the value is an array, format it using custom formatter
        if isinstance(value, np.ndarray):
            # Format each element with the custom formatter and left-pad with empty characters
            formatted_value = np.array2string(value, formatter={'float_kind': lambda x: f"{custom_formatter(x):>7}"})
        else:
            formatted_value = value
        logger.info(f'{key}: \n {formatted_value}')

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000,
            use_tta=False,
            os_weight=1.0,
            ):
    

    logger.info("Start training")
    counter = 0

    for epoch in range(epoch_start, max_epochs):
        
        logger.info(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
            use_tta=False,
            os_weight=os_weight,
        )

        results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        logger.info("TRAINING: loss_action: {:.5f}, loss_offence_severity: {:.5f}".format(loss_action, loss_offence_severity))
        prettyprint(results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train = False,
            set_name="valid",
            use_tta=use_tta,
            os_weight=os_weight,
        )

        results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        logger.info("\nVALIDATION: loss_action: {:.5f}, loss_offence_severity: {:.5f}".format(loss_action, loss_offence_severity))
        prettyprint(results)


        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
                use_tta=use_tta,
                os_weight = os_weight,
            )

        results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        logger.info("\nTEST: loss_action: {:.5f}, loss_offence_severity: {:.5f}".format(loss_action, loss_offence_severity))
        prettyprint(results)
        

        scheduler.step()

        counter += 1

        if counter > 3:
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch+1) + "_model.pth.tar")
            torch.save(state, path_aux)
        
    pbar.close()    
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
          use_tta=False,
          os_weight=1.0,
        ):
    
    scaler = GradScaler()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}
    
    for targets_offence_severity, targets_action, mvclips, action in dataloader:

        # compute output
        with autocast(enabled=True):
            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()
            outputs_offence_severity, outputs_action, attention_vector = model(mvclips)

            if(use_tta): #Test time augmentation
                if(set_name=='test' or set_name=='chall' or set_name=='valid'): #Add a flipped version and average
                    outputs_offence_severity_flipped, outputs_action_flipped, _ = model(torch.flip(mvclips, dims=[-1]))
                    outputs_offence_severity = (outputs_offence_severity + outputs_offence_severity_flipped)/2.0
                    outputs_action = (outputs_action + outputs_action_flipped)/2.0
        
        if len(action) == 1:
            preds_sev = torch.argmax(outputs_offence_severity, 0)
            preds_act = torch.argmax(outputs_action, 0)

            values = {}
            values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
            if preds_sev.item() == 0:
                values["Offence"] = "No offence"
                values["Severity"] = ""
            elif preds_sev.item() == 1:
                values["Offence"] = "Offence"
                values["Severity"] = "1.0"
            elif preds_sev.item() == 2:
                values["Offence"] = "Offence"
                values["Severity"] = "3.0"
            elif preds_sev.item() == 3:
                values["Offence"] = "Offence"
                values["Severity"] = "5.0"
            actions[action[0]] = values       
        else:
            preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
            preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

            for i in range(len(action)):
                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                if preds_sev[i].item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev[i].item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev[i].item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev[i].item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[i]] = values       

        
        if len(outputs_offence_severity.size()) == 1:
            outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
        if len(outputs_action.size()) == 1:
            outputs_action = outputs_action.unsqueeze(0)  

        with autocast(enabled=True):
            #compute the loss

            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = os_weight*loss_offence_severity + loss_action
            #import pdb; pdb.set_trace()

        if train:
            # backpropagation with AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            

        loss_total_action += float(loss_action)
        loss_total_offence_severity += float(os_weight*loss_offence_severity)
        total_loss += 1

        if pbar is not None:
            pbar.set_description(f"ALoss: {(loss_total_action)/total_loss:.4f}, OSLoss: {(loss_total_offence_severity)/total_loss:.4f}, LR: {optimizer.param_groups[-1]['lr']:.3E}") #S: {scaler.get_scale():.1f}
            pbar.update()
          
        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  
    return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss




def evaluation(dataloader, model, set_name="test", use_tta=False):
    model.eval()

    prediction_file = "predictions_" + set_name + ".json"
    data = {"Set": set_name}
    actions = {}

    with torch.no_grad():
        for _, _, mvclips, action in dataloader:
            mvclips = mvclips.cuda().float()

            # Execute model's forward pass within autocast context
            with autocast(enabled=True):
                outputs_offence_severity, outputs_action, _ = model(mvclips)

                if(use_tta):
                    if(set_name=='test' or set_name=='chall' or set_name=='valid'): #Add a flipped version and average
                        outputs_offence_severity_flipped, outputs_action_flipped, _ = model(torch.flip(mvclips, dims=[-1]))
                        outputs_offence_severity = (outputs_offence_severity + outputs_offence_severity_flipped)/2.0
                        outputs_action = (outputs_action + outputs_action_flipped)/2.0
                        
            #import pdb; pdb.set_trace()

            if len(action) == 1:
                #outputs_action[6] = -10000 #remove challenge class from action preds
                #outputs_offence_severity[-1] = -10000 #remove red card class from offence preds
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {
                    "Action class": INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                }
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {
                        "Action class": INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    }
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values

        gc.collect()
        torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(prediction_file, "w") as outfile:
        json.dump(data, outfile)
    return prediction_file

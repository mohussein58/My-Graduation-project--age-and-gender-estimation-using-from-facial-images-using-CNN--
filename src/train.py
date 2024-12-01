import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time, strftime
import torch
import os
from tqdm import tqdm
from torchvision import transforms
from random import randint

def print_training_status(epoch_count, images_seen, train_loss, val_loss,
                          elapsed_time, patience_count):
    print(
        # https://stackoverflow.com/a/8885688
        f"| {'*' * patience_count:10}{epoch_count:3.0f}",
        f"| {images_seen:13}",
        f"| {train_loss:13.4g}",
        f"| {val_loss:13.4g}",
        f"| {elapsed_time//60:10.0f}:{elapsed_time%60:02.0f} |",
    )


# TRAIN FUNCTION
def train_model(model, train_set, val_set, model_save_dir='temp_models',
                 learning_rate=0.0005, max_epochs=30, patience=3,
                loss_fn=nn.CrossEntropyLoss(),
                optim_fn=torch.optim.Adam, batch_size=32, filename_note=None,
                image_resize=None, aug_transform=None):

    model_save_dir = '../' + model_save_dir # use main folder rather than /src/
    # Init variables
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_name = (f'{model.__class__.__name__}-{model.num_classes}'
                       + ((filename_note + '_') if filename_note else '_')
                       + strftime("%d%m-%H%M") + '.pt')
    model_save_path = os.path.join(model_save_dir, model_save_name)
    optim = optim_fn(model.parameters(), lr=learning_rate)
    images_seen = 0
    best_val_loss = None
    patience_count = 0
    halt_reason = None
    checkpoint = None

    # move model to gpu for speed if available
    if torch.cuda.is_available():
        model.to('cuda')

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)

    print(f"""
TRAINING MODEL {model_save_name} WITH PARAMS:
 - Architecture: {model.__class__.__name__}
 - Learning rate: {learning_rate}
 - Optimizer: {optim.__class__.__name__}
 - Loss function: {loss_fn}
 - Other notes: {filename_note if filename_note else 'None'}

+---------------+---------------+---------------+---------------+---------------+
|         EPOCH | EXAMPLES SEEN |    TRAIN LOSS |      VAL LOSS |  ELAPSED TIME |
+---------------+---------------+---------------+---------------+---------------+
""", end='')

    start_time = time()  # Get start time to calculate training time later

    try: # For custom keyboard interrupt
        # TRAINING LOOP
        for epoch_count in range(1, max_epochs+1):
            # Train model with training set
            model.train()  # Set model to training mode
            train_loss = 0
            for images, labels in tqdm(train_loader, position=0, leave=False,
                                    desc=f'Epoch {epoch_count}') :  # iterate through batches
                if torch.cuda.is_available(): # can this be done to whole dataset instead?
                    images, labels = images.to('cuda'), labels.to('cuda')

                if image_resize is not None:
                    images = transforms.Resize((image_resize, image_resize))(images)
                if aug_transform is not None:
                    images = aug_transform(images)

                optim.zero_grad()
                outputs = model(images)  # Forward pass
                loss_fn(outputs, labels)
                loss = (
                    loss_fn(outputs, labels) if isinstance(loss_fn, nn.CrossEntropyLoss)
                    # For MSELoss, need to unsqueeze labels to match outputs
                    else loss_fn(outputs, labels.float().unsqueeze(1)) / batch_size)
                loss.backward()
                optim.step()
                train_loss += loss.item()
                images_seen += len(images)

            train_loss /= len(train_loader)

            # Test model with separate validation set
            model.eval()  # Set model to evaluation mode
            val_loss = 0
            for images, labels in tqdm(val_loader, position=0, leave=False,
                                    desc=f'Testing epoch {epoch_count}'):
                if torch.cuda.is_available(): # can this be done to whole dataset instead?
                    images, labels = images.to('cuda'), labels.to('cuda')

                if image_resize is not None:
                    images = transforms.Resize((image_resize, image_resize))(images)

                outputs = model(images)
                loss = (
                    loss_fn(outputs, labels) if isinstance(loss_fn, nn.CrossEntropyLoss)
                    else loss_fn(outputs, labels.float().unsqueeze(1)) / batch_size)
                val_loss += loss.item()
                
            val_loss /= len(val_loader)

            # print(f'val example: expected {labels[0]}, got {outputs[0]}')

            if best_val_loss is None or val_loss < best_val_loss:
                # Save model if best so far
                best_val_loss = val_loss
                checkpoint = model.state_dict()
                torch.save(checkpoint, model_save_path)
                patience_count = 0
            else:
                # learning_rate *= 0.5 # Reduce learning rate?
                # Reset optimiser momentum to encourage new exploration:
                patience_count += 1 # Increase patience count
                
            # Display epoch results
            elapsed_time = time() - start_time
            print_training_status(epoch_count, images_seen, train_loss,
                                val_loss, elapsed_time, patience_count)

            if patience_count >= patience:
                # Stop training if val loss hasn't improved for a while
                model.load_state_dict(checkpoint)
                halt_reason = 'patience'
                break

                # Or warm restart: load best model (checkpoint) and try again
                # Ommitted due to time constraints and wasn't working properly :(
                # model.load_state_dict(checkpoint)
                # optim = optim_fn(model.parameters(), lr=learning_rate)
                # patience_count = 0


    except KeyboardInterrupt:
        halt_reason = 'keyboard'
        pass
    
    print('+---------------+---------------+---------------+---------------+---------------+')
    
    if halt_reason is None:
        print(f"\nHlating training - epoch limit ({max_epochs}) reached")
    elif halt_reason == 'patience':
        print(f"\nHalting training - {patience} epochs without improvement")
    elif halt_reason == 'keyboard':
        print(f"\nHalting training - stopped by user")

    # Calculate total training time
    end_time = time()
    training_time_s = end_time - start_time
    print(
        f'\nTraining took {training_time_s//60:.0f}m {training_time_s%60:02.0f}s',
        f'({training_time_s//epoch_count}s per epoch)' if epoch_count > 0 else '')
    
    print(f"\nBest model from session saved to '{model_save_path}'\n")

    # Return best model
    return model
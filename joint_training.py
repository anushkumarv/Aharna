import torch
from torch import nn
import time

def joint_trainning(compress_net_model, query_net_model, optimizer, loss_fn, train_dl, val_dl=None, epochs=10, device='cuda'):

    print('train() called: compress net model=%s, query net mode=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(compress_net_model).__name__, type(query_net_model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    # history['val_loss'] = []
    # history['acc'] = []
    # history['val_acc'] = []

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        compress_net_model.train()
        query_net_model.train()
        train_loss         = 0.0

        for batch in train_dl:

            optimizer.zero_grad()

            cnet_x = batch[0]
            qnet_x = batch[1]
            cnet_y = compress_net_model(cnet_x)
            qnet_y = query_net_model(qnet_x)
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            y_hat = cos(cnet_y, qnet_y)
            batch_size = len(cnet_x)
            y = torch.tensor([1.0,0.0]).repeat(batch_size, 1)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss  += loss.data.item() * batch_size

        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        # if val_dl:
        #     model.eval()
        #     val_loss       = 0.0
        #     num_val_correct  = 0
        #     num_val_examples = 0

        #     for batch in val_dl:

        #         x    = batch[0].to(device)
        #         y    = batch[1].to(device)
        #         y    = torch.unsqueeze(y, 1)
        #         yhat = model(x)
        #         loss = loss_fn(yhat, y)

        #         val_loss  += loss.data.item() * x.size(0)

        #     val_loss = val_loss / len(val_dl.dataset)


        #     if epoch == 1 or epoch % 2 == 0:
        #         print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % \
        #             (epoch, epochs, train_loss, val_loss))

        history['loss'].append(train_loss)
        #     history['val_loss'].append(val_loss)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

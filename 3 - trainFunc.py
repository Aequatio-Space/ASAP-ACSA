import datetime
import time
from tqdm import tqdm
import torch
from train_utils import collector,evaluateSENT,evaluateSENT_CLS,evaluateTSV
#standard train for regression version of RP
def train_PureReg(model, dataloader, test_dataloader, loss_func, optimizer, epochs, device, save_name, logger, Reg=False):
    t0 = time.time()
    plot_loss = collector(["epoch","train_loss","val_loss"])
    plot_acc = collector(["epoch","acc"])
    model = model.to(device)
    best_loss = 100.0
    best_acc = -1
    best_epoch = -1
    for epoch in range(epochs):
        print('='*10 + f"epoch {epoch+1}" + '='*10)
        avg_loss = 0
        step = 0
        total = len(dataloader)
        for X,y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(torch.float).to(device)
            output = model(X)
            train_loss = loss_func(output.squeeze(1),y)
            avg_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1
            if step%50==0:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
                logger.info(f"Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
        avg_loss /= len(dataloader)
        bestEpochCallOut = f"Now the best loss {best_loss:<6.2f} is at Epoch {epoch + 1}"
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" Avg Train Loss of Epoch {epoch+1}: {avg_loss:>5.3f}")
        logger.info(f"Avg Train Loss of Epoch {epoch+1}:{avg_loss:>5.3f}")
        result = evaluateSENT(model, None ,test_dataloader, loss_func, device,FullLabel=False,Reg=Reg)
        if isinstance(result,tuple):
            loss,acc = result[0],result[1]
            logger.info(f"Avg Test Loss of Epoch {epoch + 1}:{loss:>5.3f} Acc: {acc}%")
        plot_loss.append(epoch+1,"epoch")
        plot_loss.append(avg_loss,"train_loss")
        plot_loss.append(loss,"val_loss")
        if not Reg:
            plot_acc.append(epoch+1,"epoch")
            plot_acc.append(acc,"acc")
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + bestEpochCallOut)
                bestCallOut = f"Best acc {best_acc:<6.2f} is at Epoch {best_epoch}"
                logger.info(bestEpochCallOut)
                torch.save(model.state_dict(), save_name + '.pt')
                plot_acc.genCSV(save_name + " acc_evo.csv")
        else:
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + bestEpochCallOut)
                bestCallOut = f"Best loss {best_loss:<6.2f} is at Epoch {best_epoch}"
                logger.info(bestEpochCallOut)
                torch.save(model.state_dict(), save_name + '.pt')
        plot_loss.genCSV(save_name + " loss_evo.csv")
        logger.info(bestCallOut)
    print("Training Completed. With " + bestCallOut + f" Total Time:{time.time()-t0:>5.2f}s")
def train(model, dataloader, test_dataloader, loss_func, optimizer, epochs, device, save_name, logger, Reg=False):
    t0 = time.time()
    plot_loss = collector(["epoch","train_loss","val_loss"])
    plot_acc = collector(["epoch","acc"])
    model = model.to(device)
    best_loss = 100.0
    best_acc = -1
    best_epoch = -1
    for epoch in range(epochs):
        print('='*10 + f"epoch {epoch+1}" + '='*10)
        avg_loss = 0
        step = 0
        total = len(dataloader)
        for X,y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(torch.float).to(device)
            output = model(X,y[:,1:])
            train_loss = loss_func(output.squeeze(1),y[:,0])
            avg_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1
            if step%50==0:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
                logger.info(f"Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
        avg_loss /= len(dataloader)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" Avg Train Loss of Epoch {epoch+1}: {avg_loss:>5.3f}")
        logger.info(f"Avg Train Loss of Epoch {epoch+1}:{avg_loss:>5.3f}")
        result = evaluateSENT(model, None ,test_dataloader, loss_func, device,FullLabel=True,Reg=Reg)
        if isinstance(result,tuple):
            loss,acc = result[0],result[1]
            logger.info(f"Avg Test Loss of Epoch {epoch + 1}:{loss:>5.3f} Acc: {acc}%")
        plot_loss.append(epoch+1,"epoch")
        plot_loss.append(avg_loss,"train_loss")
        plot_loss.append(loss,"val_loss")
        if not Reg:
            plot_acc.append(epoch+1,"epoch")
            plot_acc.append(acc,"acc")
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + bestEpochCallOut)
                bestCallOut = f"Best acc {best_loss:<6.2f}% is at Epoch {best_epoch}"
                logger.info(bestCallOut)
                torch.save(model.state_dict(), save_name + '.pt')
                plot_acc.genCSV(save_name + " acc_evo.csv")
        else:
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + bestEpochCallOut)
                bestCallOut = f"Best loss {best_loss:<6.2f} is at Epoch {best_epoch}"
                logger.info(bestCallOut)
                torch.save(model.state_dict(), save_name + '.pt')
        plot_loss.genCSV(save_name + " loss_evo.csv")
        logger.info(bestCallOut)
    print("Training Completed. With " + bestCallOut + f" Total Time:{time.time()-t0:>5.2f}s")
#train for classification version of RP
def train_CLS(model, dataloader, test_dataloader, loss_func, optimizer, epochs, device, save_name, logger, Reg=False):
    t0 = time.time()
    plot_loss = collector(["epoch","train_loss","val_loss"])
    plot_acc = collector(["epoch","acc"])
    model = model.to(device)
    best_loss = 100.0
    best_acc = -1
    best_epoch = -1
    for epoch in range(epochs):
        print('='*10 + f"epoch {epoch+1}" + '='*10)
        avg_loss = 0
        step = 0
        total = len(dataloader)
        for X,y in tqdm(dataloader):
            X = X.to(device)
            y = (y-1).to(device).to(torch.long)
            #Historical Reason:dummy y.
            output = model(X)
            train_loss = loss_func(output,y[:,0])
            avg_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1
            if step%50==0:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
                logger.info(f"Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
        avg_loss /= len(dataloader)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" Avg Train Loss of Epoch {epoch+1}: {avg_loss:>5.3f}")
        logger.info(f"Avg Train Loss of Epoch {epoch+1}:{avg_loss:>5.3f}")
        result = evaluateSENT_CLS(model, None ,test_dataloader, loss_func, device,FullLabel=True)
        if isinstance(result,tuple):
            loss,acc = result[0],result[1]
            logger.info(f"Avg Test Loss of Epoch {epoch + 1}:{loss:>5.3f} Acc: {acc}%")
        plot_loss.append(epoch+1,"epoch")
        plot_loss.append(avg_loss,"train_loss")
        plot_loss.append(loss,"val_loss")
        if not Reg:
            plot_acc.append(epoch+1,"epoch")
            plot_acc.append(acc,"acc")
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                bestCallOut = f" Best acc {best_acc:<6.2f}% is at Epoch {best_epoch}"
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + bestCallOut)
                logger.info(bestCallOut)
                torch.save(model.state_dict(), save_name + '.pt')
            plot_acc.genCSV(save_name + " acc_evo.csv")
        else:
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                bestCallOut = f" Best loss {best_loss:<6.2f}% is at Epoch {best_epoch}"
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + bestCallOut)
                logger.info(bestCallOut)
                torch.save(model.state_dict(), save_name + '.pt')
        plot_loss.genCSV(save_name + " loss_evo.csv")
        logger.info(bestCallOut)
    print("Training Completed. With " + bestCallOut + f" Total Time:{time.time()-t0:>5.2f}s")
def train_ASPECT(model, dataloader, test_dataloader, loss_func, optimizer, epochs, device, save_name, logger,join = False):
    t0 = time.time()
    plot_loss = collector(["epoch","train_loss","val_loss"])
    plot_acc = collector(["epoch","acc"])
    model = model.to(device)
    best_acc = -1
    best_epoch = -1
    for epoch in range(epochs):
        print('='*10 + f"epoch {epoch+1}" + '='*10)
        avg_loss = 0
        step = 0
        total = len(dataloader)
        for X,keyword,y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(torch.long).to(device)
            keyword = keyword.to(device)
            output = model(X,keyword)
            if join:
                for i in range(y.shape[0]):
                    if y[i]==1:
                        y[i] = torch.randint(1,3,(1,))
                    elif y[i]==2:
                        y[i] = torch.randint(3,5,(1,))
            train_loss = loss_func(output,y)
            avg_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1
            if step%50==0:
                print(f"Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
                logger.info(f"Step {step}/{total} Batch Loss: {train_loss.item():>5.3f}")
        avg_loss /= len(dataloader)
        print(f"Avg Train Loss of Epoch {epoch+1}: {avg_loss:>5.3f}")
        logger.info(f"Avg Train Loss of Epoch {epoch+1}:{avg_loss:>5.3f}")
        result = evaluateTSV(model, None ,test_dataloader, loss_func, device)
        plot_loss.append(epoch+1,"epoch")
        plot_loss.append(avg_loss,"train_loss")
        plot_loss.append(result[0],"val_loss")
        plot_acc.append(epoch+1,"epoch")
        plot_acc.append(result[1],"acc")
        logger.info(f"Acc: {result[1]:<6.2f}%")
        if result[1]>best_acc:
            best_acc = result[1]
            best_epoch = epoch + 1
            bestCallOut = f"Best acc {best_acc:<6.2f}% is at Epoch {best_epoch}"
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + bestCallOut)
            logger.info(bestCallOut)
            torch.save(model.state_dict(), save_name + '.pt')
            plot_acc.genCSV(save_name + " acc_evo.csv")
        plot_loss.genCSV(save_name + " loss_evo.csv")
    print("Training Completed. With " + bestCallOut + f" Total Time:{time.time() - t0:>5.2f}s")

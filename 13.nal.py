# -*- coding: utf-8 -*-
# Nevronske mreže: 1. in 2. naloga

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from data_load_nal13 import *
from timeit import default_timer as timer
import os
from ProcessPriority import *
from concurrent.futures import ProcessPoolExecutor

# Add the directory containing your desired OpenMP library to PATH
os.environ['PATH'] = 'C:/Users/All Users/anaconda3/Library/bin/libiomp5md.dll;' + os.environ['PATH']
# Optionally, prevent conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
HighPriority()
plt.style.use('aljaz')
np.printoptions(linewidth=np.nan)

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from neural_func import *

nthd = 4 #Torch, OpenBlas and MKL will use nthd threads
# torch.set_num_threads(nthd) #paralelisation for single operation
# torch.set_num_interop_threads(nthd) #multiple operations, dafault: num. cores
os.environ['MKL_NUM_THREADS'] = str(nthd)
os.environ['OMP_NUM_THREADS'] = str(nthd)
os.environ['OPENBLAS_NUM_THREADS'] = str(nthd)

############################
if __name__ == '__main__':
    print(f'intra-op: {torch.get_num_threads()}, inter-op: {torch.get_num_interop_threads()}')

    # #   manual import data za prikaz
    # t1=timer()
    # images=[]
    # labels=[]
    # for image, label in datareader("./podatki/train-images-idx3-ubyte", "./podatki/train-labels-idx1-ubyte"): #learn
    #     image = image/max(image) #scale 0-1
    #     images.append(image)
    #     labels.append(label)
    #     # print("Loaded image with label:", label) #zelo upočasni!!!
    # images_l=np.array(images)
    # labels_l=np.array(labels)
    # t2=timer()
    # print(f'manual: t_load = {(t2-t1):.1f} s')

    # slike=10 #slik po osi
    # fig,ax=plt.subplots(slike, slike, figsize=(8,16))
    # j=k=0
    # for i in range(slike*slike):
    #     ax[j,k].imshow(images_l[i].reshape((28,28)), cmap='binary') #784 px
    #     ax[j,k].axis('off')
    #     # ax[j,k].set_xticks([])
    #     # ax[j,k].set_yticks([])
    #     ax[j,k].set_title(labels[i])
    #     j+=1
    #     if j == slike:
    #         j=0
    #         k+=1
    # plt.tight_layout()
    # # plt.savefig('./slike/train_images.pdf')
    # plt.show()

    images=[]
    labels=[]
    for image, label in datareader("./podatki/t10k-images-idx3-ubyte", "./podatki/t10k-labels-idx1-ubyte"): #test
        image = image/max(image) #scale 0-1
        images.append(image)
        labels.append(label)
    images_t=np.array(images)
    labels_t=np.array(labels)
    # fig,ax=plt.subplots(slike, slike, figsize=(8,16))
    # j=k=0
    # for i in range(slike*slike):
    #     ax[j,k].imshow(images_t[i].reshape((28,28)), cmap='binary') #784 px
    #     ax[j,k].axis('off')
    #     ax[j,k].set_title(labels[i])
    #     j+=1
    #     if j == slike:
    #         j=0
    #         k+=1
    # plt.tight_layout()
    # # plt.savefig('./slike/test_images.pdf')
    # plt.show()

    # plt.hist(labels_l,label='učenje',histtype='step', align='left',density=True)
    # plt.hist(labels_t,label='test',histtype='step', align='left',density=True)
    # plt.xticks(np.linspace(0,9,11), labels=np.append(np.linspace(0,9,10,dtype=int),[None]))
    # plt.xlabel('števke')
    # plt.ylabel('pogostost')
    # plt.legend()
    # plt.tight_layout()
    # # plt.savefig('./slike/MNIST_hist.pdf')
    # plt.show()

    ##############################
    #   torchvision import data -> hitreje!
    batch_size = 64 #how many samples are processed together in a single forward/backward pas (2**N!) -> hitreje?

    t1=timer()
    # Define a transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts images to PyTorch tensors and scales them to [0, 1]
        #transforms.Normalize((0.5,), (0.5,))  # Optionally normalize to [-1, 1] range
    ])
    # Load the dataset using torchvision's MNIST class
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    t2=timer()
    print(f'torch: t_load = {(t2-t1):.1f} s')
    # print(len(trainloader.dataset), len(testloader.dataset)) #train set: 60 000, test set: 10 000

    # images, labels = next(iter(trainloader))
    # images = torch.tensor(images, dtype=torch.float32)#single prec.
    # labels = torch.tensor(labels, dtype=torch.long)#integer 64bit = torch.int64
    # print(images.shape)
    # print(labels.shape)
    # print(len(trainloader)*batch_size)#ni nujno vse polno!

    # plt.imshow(images[slika].reshape((28,28)), cmap='binary') #drug vrstni red, vsakič drugače!!
    # plt.axis('off')
    # plt.show()

    #########################
    #   učenje mreže
    Hs = 256 #size of hidden layer
    lr = 0.1 #parameter hitrosti učenja
    N_epoch = 100 #number of epochs for training

    ########## 1. določanje Lr
    # lr_list=[0.001,0.01,0.1,1]
    # tasks = [(trainloader, testloader, Hs, lr, N_epoch) for lr in lr_list]
    # max_workers=len(lr_list)
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(learn, task) for task in tasks]
    #     results = [future.result() for future in futures]
    # delez_prav=np.zeros((len(lr_list),N_epoch))
    # napaka=np.zeros((len(lr_list),N_epoch))
    # models=[]
    # loss_f_list=[]
    # for i, (model, loss_f, correct, test_loss,t) in enumerate(results):
    #     delez_prav[i] = correct
    #     napaka[i] = test_loss
    #     models.append(model)
    #     loss_f_list.append(loss_f)
    
    # epohe=np.arange(1,N_epoch+1,1,dtype=int)
    # c_list=[f'C{i}' for i in range(len(lr_list))]
    # fig,ax=plt.subplots(figsize=(6,4))
    # ax1=plt.twinx(ax)
    # for i in range(len(lr_list)):
    #     c=c_list[i]
    #     ax.plot(epohe, 100*delez_prav[i], '-', c=c, label=f'{lr_list[i]}') #label='delež pravilnih')
    #     ax1.plot(epohe, napaka[i], '--', c=c) # label='skupna napaka')
    #     # plt.axhline(correct[-1], c=c, ls=':')
    #     # plt.axhline(test_loss[-1], c=c, ls=':')
    #     # plt.text(epohe[-1], correct[-1]-0.1, f'{correct[-1]:.3f}', c=c)
    #     # plt.text(epohe[-1], test_loss[-1]+0.1, f'{test_loss[-1]:.3f}', c=c)
    # ax.legend(title='hitrost učenja',loc='upper left',bbox_to_anchor=(0,1.3),ncol=len(lr_list))
    # ax.set_ylim(0,100)
    # ax1.set_yscale('log')
    # ax.set_xlabel('epohe')
    # ax.set_ylabel('polna črta: delež pravilnih [\%]')
    # ax1.set_ylabel('prekinjena črta: loss function skupna napaka')
    # plt.tight_layout()
    # # plt.savefig('./slike/learning_curve.pdf')
    # plt.show()

    ################## 2. določanje velikost skrite plasti
    # Hs_list=np.append(np.linspace(1,10,10, dtype=int), np.append(np.linspace(11,1e2,20, dtype=int)[:-2], np.logspace(2,3,10, dtype=int)))
    # N_epoch=10
    # lr=0.1
    # tasks = [(trainloader, testloader, Hs, lr, N_epoch) for Hs in Hs_list]
    # max_workers=min([len(Hs_list),19])
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(learn, task) for task in tasks]
    #     results = [future.result() for future in futures]
    # delez_prav=np.zeros((len(Hs_list),N_epoch))
    # napaka=np.zeros((len(Hs_list),N_epoch))
    # models=[]
    # loss_f_list=[]
    # for i, (model, loss_f, correct, test_loss) in enumerate(results):
    #     delez_prav[i] = correct
    #     napaka[i] = test_loss
    #     models.append(model)
    #     loss_f_list.append(loss_f)
    
    # fig,ax=plt.subplots(figsize=(6,4))
    # ax1=plt.twinx(ax)
    # ax.plot(Hs_list, 100*delez_prav[:,-1], 'C0-',label='končna pravilnost')
    # # ax.plot(Hs_list, 100*np.max(delez_prav,axis=1), '--', label='max pravilnost')
    # ax1.plot(Hs_list, napaka[:,-1], 'C1-', label='končna skupna napaka')
    # # ax1.plot(Hs_list, 100*np.min(napaka,axis=1), 'C2--', label='max skupna napaka')
    # # ax.set_ylim(10,100)
    # ax1.set_yscale('log')
    # ax.set_xlabel('velikost skrite plasti')
    # ax.set_ylabel('delež pravilnih [\%]',c='C0')
    # ax.tick_params(labelcolor='C0')
    # ax1.set_ylabel('loss function skupna napaka',c='C1')
    # ax1.tick_params(labelcolor='C1')
    # plt.tight_layout()
    # plt.savefig('./slike/velikost_skrite_plasi_.pdf')
    # plt.show()

    #################### 3. Več skritih plasti
    # lr, N_epoch = 0.001, 20
    # Nl_list=np.linspace(1,10,10, dtype=int)

    # tasks = [(trainloader, testloader, Nl, lr, N_epoch) for Nl in Nl_list]
    # max_workers=min([len(Nl_list),18])
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(learn, task) for task in tasks]
    #     results = [future.result() for future in futures]
    # delez_prav=np.zeros((len(Nl_list),N_epoch))
    # napaka=np.zeros((len(Nl_list),N_epoch))
    # models=[]
    # loss_f_list=[]
    # time=[]
    # layers=[]
    # for i, (model, loss_f, correct, test_loss, t) in enumerate(results):
    #     delez_prav[i] = correct
    #     napaka[i] = test_loss
    #     models.append(model)
    #     loss_f_list.append(loss_f)
    #     time.append(t)
    #     layers.append(model.layer_sizes)

    # epohe=np.arange(1,N_epoch+1,1,dtype=int)
    # c_list=[f'C{i}' for i in range(len(Nl_list))]
    # fig,ax=plt.subplots(figsize=(6,4))
    # ax1=plt.twinx(ax)
    # for i in range(0,len(Nl_list),2):
    #     c=c_list[i]
    #     ax.plot(epohe, 100*delez_prav[i], '-', c=c, label=f'{Nl_list[i]}') #label='delež pravilnih')
    #     ax1.plot(epohe, napaka[i], '--', c=c) # label='skupna napaka')
    # ax.legend(title='število skritih plasti',loc='upper left',bbox_to_anchor=(0,1.2),ncol=len(Nl_list))
    # # ax.set_ylim(0,100)
    # ax1.set_yscale('log')
    # ax.set_xlabel('epohe')
    # ax.set_ylabel('polna črta: delež pravilnih [\%]')
    # ax1.set_ylabel('prekinjena črta: loss function skupna napaka')
    # plt.tight_layout()
    # plt.savefig('./slike/vec_plasti.pdf')
    # plt.show()

    # plt.plot(Nl_list, delez_prav[:,-1], 'k.-')
    # plt.xlabel('št. skritih plasti')
    # plt.ylabel('končni delež pravilnih razvrstitev')
    # plt.savefig('./slike/vec_plasti_koncno.pdf')
    # plt.show()

    # plt.plot(Nl_list, time, 'k.-')
    # plt.xlabel('št. skritih plasti')
    # plt.ylabel('čas učenja [s]')
    # plt.savefig('./slike/vec_plasti_time.pdf')
    # plt.show()

    # for i in range(len(Nl_list)):
    #     plt.plot(np.linspace(0,1,len(layers[i])), layers[i], '.-', label=f'{Nl_list[i]}')
    # plt.xlabel('skalirane plasti')
    # plt.ylabel('porazdelitev nevronov po plasteh')
    # plt.savefig('./slike/vec_plasti_nevroni.pdf')
    # plt.show()

    ################## 4. konvolucijska mreža
    # lr, N_epoch, Hs, kernel_size = 0.1,10,32,3
    # model, loss_f, correct, test_loss,t = learn((trainloader, testloader, Hs, lr, N_epoch))

    # print(max(correct)*100, correct[-1]*100)
    # epohe=np.arange(1,N_epoch+1,1,dtype=int)
    # fig,ax=plt.subplots(figsize=(6,4))
    # ax1=plt.twinx(ax)
    # ax.plot(epohe, 100*correct, '-', c='C0', label='delež pravilnih')
    # ax1.plot(epohe, test_loss, '--', c='C1', label='skupna napaka')
    # ax.axhline(100*correct[-1], c='C0', ls=':')
    # ax1.axhline(test_loss[-1], c='C1', ls=':')
    # ax.text(epohe[-1]*0.8, 100*correct[-1]-5, f'{(100*correct[-1]):.3f}', c='C0')
    # ax1.text(epohe[-1]+1, test_loss[-1], f'{test_loss[-1]:.3f}', c='C1')
    # # ax.set_ylim(0,100)
    # ax1.set_yscale('log')
    # ax.set_xlabel('epohe')
    # ax.set_ylabel('delež pravilnih [\%]', c='C0')
    # ax.tick_params(axis='y', labelcolor='C0')
    # ax1.set_ylabel('loss function skupna napaka', c='C1')
    # ax1.tick_params(axis='y', labelcolor='C1')
    # plt.tight_layout()
    # plt.savefig('./slike/konvolucijska.pdf')
    # plt.show()
    # # torch.save(model.state_dict(), f'./neural_models/konvolucijska_Hs{Hs}_Lr{lr}_N_epoh{N_epoch}.pt') #save model

    ################## 5. najbolje učenje -> rezultati
    lr, N_epoch, Hs = 0.1,20,256 # SGD optimizer
    # lr, N_epoch, Hs = 0.001,10,256 # ADAM optimizer
    # model, loss_f, correct, test_loss,t = learn((trainloader, testloader, Hs, lr, N_epoch))

    # print(max(correct)*100, correct[-1]*100)
    # epohe=np.arange(1,N_epoch+1,1,dtype=int)
    # fig,ax=plt.subplots(figsize=(6,4))
    # ax1=plt.twinx(ax)
    # ax.plot(epohe, 100*correct, '-', c='C0', label='delež pravilnih')
    # ax1.plot(epohe, test_loss, '--', c='C1', label='skupna napaka')
    # ax.axhline(100*correct[-1], c='C0', ls=':')
    # ax1.axhline(test_loss[-1], c='C1', ls=':')
    # ax.text(epohe[-1]*0.8, 100*correct[-1]-5, f'{(100*correct[-1]):.3f}', c='C0')
    # ax1.text(epohe[-1]+1, test_loss[-1], f'{test_loss[-1]:.3f}', c='C1')
    # ax.set_ylim(0,100)
    # ax1.set_yscale('log')
    # ax.set_xlabel('epohe')
    # ax.set_ylabel('delež pravilnih [\%]', c='C0')
    # ax.tick_params(axis='y', labelcolor='C0')
    # ax1.set_ylabel('loss function skupna napaka', c='C1')
    # ax1.tick_params(axis='y', labelcolor='C1')
    # plt.tight_layout()
    # plt.show()
    # # torch.save(model.state_dict(), f'./neural_models/SGD_Hs{Hs}_Lr{lr}_N_epoh{N_epoch}.pt') #save model
    # # torch.save(model.state_dict(), f'./neural_models/ADAM_Hs{Hs}_Lr{lr}_N_epoh{N_epoch}.pt') #save model

    ###############################
    # load learned model
    lr, N_epoch, Hs = 0.1,20,256 # SGD optimizer
    model = SimpleNN(Hs)
    model.load_state_dict(torch.load(f'./neural_models/SGD_Hs{Hs}_Lr{lr}_N_epoh{N_epoch}.pt', weights_only=True))
    # lr, N_epoch, Hs = 0.001,10,256 # ADAM optimizer
    # model.load_state_dict(torch.load(f'./neural_models/ADAM_Hs{Hs}_Lr{lr}_N_epoh{N_epoch}.pt', weights_only=True))
    # model = ConvNN(32,3)
    # model.load_state_dict(torch.load(f'./neural_models/konvolucijska_Hs32_Lr0.1_N_epoh10.pt', weights_only=True))

    model.eval()
    loss_f = nn.CrossEntropyLoss(reduction='mean') #GPT predlaga - ne treba spreminjati dimenzij!!
    c,l=test(testloader, model, loss_f)
    print(f"Accuracy: {(100*c):>0.1f}%, Avg loss (error from loss func): {l:>8f}")
    
    ############ vpliv px na odločitev
    def vpliv(number, ax1,ax2,ax3):
        sign=True
        i=1100 #določi začetni index
        while sign: #find first label with desired number
            i+=1
            if labels[i] == number:
                sign=False
        img,napoved,out,loss = gradient(model, loss_f, number=labels[i], input=images[i])
        if napoved == number: c='green'
        else: c='red'
        ax1.imshow(images[i].reshape((28,28)), cmap='binary', aspect='equal')
        ax1.set_title(f'{napoved} ({number})', c=c)
        ax1.axis('off')
        ax2.imshow(img, cmap='binary', aspect='equal')
        ax2.axis('off')
        ax3.imshow(out, cmap='binary', aspect='equal')
        ax3.set_title(f'loss {loss:.1E}')
        ax3.set_xticks([])
        ax3.set_yticks([])

    # fig,ax = plt.subplots(3,10, figsize=(16,6))
    # for n in range(10):
    #     vpliv(n, ax[0,n], ax[1,n], ax[2,n])
    # plt.suptitle(f'vpliv pixlov na odločitev')
    # plt.tight_layout()
    # # plt.savefig('./slike/vpliv_px.pdf')
    # plt.show()

    ########### test napak
    test_loss, calc_val, correct, n_false, predicted = bench(testloader, model, loss_f)

    #najpogostejše napake, zamenjave
    zamenjave=[]
    napovedi=[]
    grid = np.zeros((10,10))
    for i in range(len(calc_val)):
        if correct[i] == False:
            zamenjave.append([calc_val[i],labels[i]]) #oblika (napovedana, prava) vrednost
            napovedi.append(predicted[i])
            grid[calc_val[i], labels[i]] +=1
    zamenjave=np.transpose(zamenjave)
    unique,  counts = np.unique(zamenjave[1,:], return_counts=True)
    print('največkrat zamnejana (izračun): ', unique[np.argmax(counts)])
    unique,  counts = np.unique(zamenjave[0,:], return_counts=True)
    print('največkrat zamnejana z (pravilna): ', unique[np.argmax(counts)])

    row_sums = grid.sum(axis=0)
    col_sums = grid.sum(axis=1)
    cmap = plt.get_cmap('binary')
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

    ax_main = plt.subplot(gs[1, 0])
    im = ax_main.imshow(grid.T, cmap=cmap, origin='lower')
    ax_main.set_xticks(np.linspace(0, 9, 10))
    ax_main.set_yticks(np.linspace(0, 9, 10))
    ax_main.set_xlabel('izračunana števka')
    ax_main.set_ylabel('pravilna števka')

    # Column sums plot (top-left)
    norm_col = Normalize(vmin=0, vmax=col_sums.max())
    colors_col = cmap(norm_col(col_sums))
    ax_col_sum = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_col_sum.bar(np.arange(10), col_sums, color=colors_col)
    ax_col_sum.set_xlim(-0.5, 9.5)
    ax_col_sum.xaxis.set_visible(False)
    ax_col_sum.yaxis.set_visible(True)
    ax_col_sum.set_ylabel('vsota stolpcev')
    ax_col_sum.spines['top'].set_visible(False)
    ax_col_sum.spines['right'].set_visible(False)
    ax_col_sum.spines['bottom'].set_visible(False)

    # Row sums plot (bottom-right)
    norm_row = Normalize(vmin=0, vmax=row_sums.max())
    colors_row = cmap(norm_row(row_sums))
    ax_row_sum = plt.subplot(gs[1, 1], sharey=ax_main)
    ax_row_sum.barh(np.arange(10), row_sums, color=colors_row)
    ax_row_sum.set_ylim(-0.5, 9.5)
    ax_row_sum.xaxis.set_visible(True)
    ax_row_sum.yaxis.set_visible(False)
    ax_row_sum.set_xlabel('vsota vrstic')
    ax_row_sum.spines['top'].set_visible(False)
    ax_row_sum.spines['right'].set_visible(False)
    ax_row_sum.spines['left'].set_visible(False)
    
    cbar = fig.colorbar(im, ax=ax_row_sum, shrink=0.9, label='pogostost')
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_main.grid(False)
    ax_col_sum.grid(False)
    ax_row_sum.grid(False)
    plt.tight_layout()
    # plt.savefig('./slike/zamenjave.pdf')
    # plt.show()

    plt.hist(zamenjave[0,:],label='izračunana števka',histtype='step', align='left')
    plt.hist(zamenjave[1,:],label='pravilna števka',histtype='step', align='left')
    plt.xticks(np.linspace(0,9,11), labels=np.append(np.linspace(0,9,10,dtype=int),[None]))
    plt.xlabel('števke')
    plt.ylabel('pogostost')
    plt.legend()
    plt.grid(False)
    # plt.savefig('./slike/zamenjave_hist.pdf')
    # plt.show()

    max_false=10*2 #max št vrstic slik po 10
    max_false = min([int(n_false),int(max_false)])
    fig,ax=plt.subplots((max_false//10)*2, 10, figsize=(10*2,(max_false//20)*2*2))
    i=0 #start index
    c=r=nf=0
    while nf < max_false:
        if correct[i] == False:
            # ax[r,c].set_title(f'$\\textcolor{{red}}{{{calc_val[i]}}} (\\textcolor{{green}}{{{labels_t[i]}}})$')
            ax[r, c].text(0.5, -0.2, f'{calc_val[i]}', color='red', fontsize=12, ha='center', va='bottom', transform=ax[r, c].transAxes)
            ax[r, c].text(0.7, -0.2, f'({labels_t[i]})', color='green', fontsize=12, ha='center', va='bottom', transform=ax[r, c].transAxes)
            ax[r,c].imshow(images[i].reshape((28,28)), cmap='binary', aspect='equal')
            ax[r,c].axis('off')
            ax[r+1,c].imshow([napovedi[nf]], cmap='binary', aspect='equal')
            ax[r+1,c].set_xticks([])
            ax[r+1,c].set_yticks([])
            c+=1
            nf+=1
            if c == 10:
                c=0#column
                r+=2#row
        i+=1
    plt.tight_layout()
    # plt.savefig(f'./slike/napake.pdf')
    # plt.show()

    ################## naključni vhod
    N=10000
    random_data = torch.rand((N,1,28,28))
    random_data = torch.normal(1,1,(N,1,28,28)) #gauss
    predict = model(random_data)
    
    data_im=random_data.detach().numpy().reshape((N,28,28))
    predict = predict.detach().numpy()
    val=predict.argmax(1)

    N_max=10
    fig,ax=plt.subplots(2, N_max, figsize=(N_max*2,4))
    i=n=0
    r=c=0
    vals=[]
    while n<N_max and i<N:
        if val[i] not in vals:
            ax[r+1,c].set_title(f'{val[i]}')
            ax[r,c].imshow(data_im[i], cmap='binary', aspect='equal')
            ax[r,c].axis('off')
            ax[r+1,c].imshow([predict[i]], cmap='binary', aspect='equal')
            ax[r+1,c].set_xticks([])
            ax[r+1,c].set_yticks([])
            c+=1
            n+=1
            vals.append(val[i])
        i+=1
    if n<N_max:
        for i in range(N_max-n):
            i=-i #od zadaj
            ax[r+1,c].set_title(f'{val[i]}')
            ax[r,c].imshow(data_im[i], cmap='binary', aspect='equal')
            ax[r,c].axis('off')
            ax[r+1,c].imshow([predict[i]], cmap='binary', aspect='equal')
            ax[r+1,c].set_xticks([])
            ax[r+1,c].set_yticks([])
            c+=1
    plt.suptitle('naključni vhodi')
    plt.tight_layout()
    # plt.savefig(f'./slike/naklj_data.pdf')
    # plt.show()

    unique,  counts = np.unique(val, return_counts=True)
    print(counts, unique)
    plt.hist(val, color='gray', bins=10, align='left')
    plt.xticks(np.linspace(0,9,11), labels=np.append(np.linspace(0,9,10,dtype=int),[None]))
    plt.xlim(-1,9)
    plt.xlabel('števke')
    plt.ylabel('pogostost')
    plt.yscale('log')
    plt.title('razvrščanje naključnih vhodov')
    plt.grid(False)
    plt.tight_layout()
    # plt.savefig('./slike/naklj_data_hist.pdf')
    # plt.show()
    plt.close()

    ################ 2. naloga: deep dream
    def dream(number, ax):
        img, loss = deep_dream(model, loss_f, number, N_iter=5000, lr=0.1)
        ax.imshow(img, cmap='binary', aspect='equal')
        ax.set_title(f'{number}')
        ax.axis('off')
        ax.text(0.2, -0.2, f'loss {loss:.1E}', transform=ax.transAxes)

    fig,ax = plt.subplots(1,10, figsize=(16,3))
    for n in range(10):
        dream(n, ax[n])
    plt.suptitle(f'idealne števke')
    plt.tight_layout()
    # plt.savefig('./slike/deep_dream.pdf')
    plt.show()
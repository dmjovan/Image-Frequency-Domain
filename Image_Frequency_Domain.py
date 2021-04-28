# OVAJ KOD SE NALAZI I U .ipynb skripti
# ali je prevelik fajl pa se ne moze videti
# isti kod je i u ovoj skripti, samo bez medjurezultata

# ucitavanje biblioteka potrebnih za rad
from __future__ import print_function
from pylab import *

%matplotlib inline

import skimage 
from skimage  import color, filters, exposure, img_as_float
from skimage.restoration import denoise_bilateral

import scipy 
from scipy import ndimage

import numpy as np

import time
from time import time

folder_path = 'sekvence/' # root folder za slike

#####################################################################

def gaussian_low_pass_filter(M, N, sigma):
    
    """
    Opis: 
        Funkcija formira Gausov niskofrekventni filtar sa 
        definisanom standardnom devijacijom (sigma), dimenzija MxN.
        Maska filtra predstavlja Gausov filtar u frekencijskom domenu,
        pri cemu je spektar centriran. 
        
    Parametri:
        M, N - dimenzije slike koja se filtrira, pa i filtra
        sigma - standardna devijacija filtra
    
        
    Funkcija vraca masku Gausovog niskofrekventnog filtra u frekvencijskom domenu.
    
    """
    
    # formiranje centriranih osa za filtar
    if (M%2 == 0):
        y = np.arange(0,M) - M/2 + 0.5 
    else:
        y = np.arange(0,M) - (M-1)/2
    
    if (N%2 == 0):
        x = np.arange(0,N) - N/2 + 0.5 
    else:
        x = np.arange(0,N) - (N-1)/2
        
    X,Y = meshgrid(x,y)
    
    # formiranje matrice rastojanja od centra filtra
    D = np.sqrt(np.square(X) + np.square(Y))
    
    # formiranje maske Gausovog 2D filtra u frekvencijskom domenu
    filter_mask = np.exp(-np.square(D)/(2*np.square(sigma)))
    
    return filter_mask
#####################################################################

# ucitavanje slike 
img_name = 'girl_ht.tif';
img = img_as_float(imread(folder_path + img_name))

# ocitavanje i ispisivanje dimenzija slike
[M, N] = shape(img)

print('Dimenzije slike su: \ndim = ' + str(M) + 'x' + str(N))

# prikaz ucitane slike
figure(figsize = (8, 8), dpi = 120);
imshow(img ,cmap = 'gray');
plt.title('Ulazna slika')
plt.axis('off')
plt.show()
#####################################################################

# prelazak u frekvencijski domen
Fimg = fftshift(fft2(img)); # spektar ulazne slike

Fimg_mag = log(1+abs(Fimg)) # amplitudski spektar slike, i to skalirani sa log funkcijom
Fimg_ang = angle(Fimg) # fazni spektar slike

# prikaz amplitudskog i faznog spektra slike
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 20), dpi = 120)
ax = axes.ravel()

ax[0].imshow(Fimg_mag, cmap = 'gray')
ax[0].set_title('Amplitudski spektar ulazne slike', fontsize = 16)
ax[0].set_axis_off()

ax[1].imshow(Fimg_ang, cmap = 'gray')
ax[1].set_title('Fazni spektar ulazne slike', fontsize = 16)
ax[1].set_axis_off()
#####################################################################

# pokusaj da se napravi Gausov low-pass filtar koji ce da filtrira samo centralni deo spektra, oko DC komponente
sigma = 30 # standardna devijacija kojom se obuhvata samo centralni deo spektra slike
H_lp =  gaussian_low_pass_filter(M, N, sigma) # spektar low-pass filtra

G_lp = Fimg*H_lp # mnozenje spekatara filtra i slike - frekvencijsko filtriranje

g = real(ifft2(ifftshift(G_lp)))
g[g<0] = 0
g[g>1] = 1

img_rescaled = exposure.rescale_intensity(g) # poboljsanje kontrasta

window_size = 13
sigma_color = 2
sigma_spatial = 5

# bilateralni filtar za odsumljavanje - sigma_color = 2, sigma_spatial = 5
# bilateralni filtar je upotrebljen i da bi se fino definisali prelazi, ivice
img_bilateral = denoise_bilateral(img_rescaled, window_size, sigma_color, sigma_spatial) 
#####################################################################

# prikaz rezultata dobijenih filtriranjem niskofrekventnim Gausovim filtrom 
fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (20, 20), dpi = 120)
ax = axes.ravel()

ax[0].imshow(log(1+abs(H_lp)), cmap = 'gray')
ax[0].set_title('Amplitudski spektar Gausovog Low-Pass fitra, sigma = ' + str(sigma), fontsize = 16)
ax[0].set_axis_off()

ax[1].imshow(log(1+abs(G_lp)), cmap = 'gray')
ax[1].set_title('Amplitudski spektar filtrirane slike' , fontsize = 16)
ax[1].set_axis_off()

ax[2].imshow(img, cmap = 'gray')
ax[2].set_title('Originalna slika', fontsize = 16)
ax[2].set_axis_off()

ax[3].imshow(g, cmap = 'gray')
ax[3].set_title('Dobijena filtrirana slika', fontsize = 16)
ax[3].set_axis_off()

ax[4].imshow(img_rescaled, cmap = 'gray')
ax[4].set_title('Skalirana filtrirana slika', fontsize = 16)
ax[4].set_axis_off()

ax[5].imshow(img_bilateral, cmap = 'gray')
ax[5].set_title('Bilateralno-filtrirana dobijena slika \n sigma_c = ' + str(sigma_color) + ', sigma_s = ' + str(sigma_spatial) + ', window = ' + str(window_size), fontsize = 16)
ax[5].set_axis_off()

plt.tight_layout()
plt.show()
#####################################################################

# postoji po 15 vidljivih Dirakovih impulsa u svakom redu i u svakoj vrsti, ali i jos po 2 na krajevima slike
# to znaci da ce nam trebati (17*17 - 1) notch filtar; rastojanja se po x-osi menjaju na svakih 155 piksela,
# a po y-osi na svakih 75 piksela 
# minus 1 potice od Dc komponente koja se nalazi na sredini slike i ne treba da se filtrira

# slika ispod pokazuje upravo ublizeno gornji levi deo spektra slike

figure(figsize = (8, 8), dpi = 120);
imshow(Fimg_mag[0:100, 0:200] ,cmap = 'gray'); # po vertikalnoj osi ide po 75 piksela, a po ,horizontalnoj osi ide po 155 piksela
#####################################################################

def gaussian_notch_reject_filter(M, N, C, radius):
    
    """
    Opis: 
        Funkcija formira notch filtar sa centrima definisanim u 
        nizu C i njhovim komplementima. Niz C sadrzi centre koji se nalaze u levom
        delu slike podeljenoj po dijagonali. DC komponenta se na nalazi medju njima. 
        Formira se notch-pass filtar za svaki element iz niza C i za svaki njegov 
        komplement, pa se nakon toga formira ukupni filtar kao 1-filtar, jer 
        funkcija sluzi da potisne komponente iz niza C.
        
    Parametri:
        M, N - dimenzije slike koja se filtrira, pa i filtra
        C - niz sa koordinatama centara koje treba filtrirati
        radius - zapravo standarna devijacija Gausovog filtra 
    
        
    Funkcija vraca masku sa Gausovim notch-reject filtrima.
    
    """
    
    # odredjivanje broja komponenti koje treba filtirati
    N_filters = len(C)
    
    # inicijalizacija filtra
    filter_mask = zeros([M, N])
    
    # formiranje i centriranje osa za filtar
    if (M%2 ==0):
        y = np.arange(0,M) - M/2 + 0.5
    else:
        y = np.arange(0,M) - (M-1)/2
        
    if (N%2 ==0):
        x = np.arange(0,N) - N/2 + 0.5
    else:
        x = np.arange(0,N) - (N-1)/2
        
    X,Y = meshgrid(x, y)
    
    # petlja za svaki element iz niza C,
    # tj. za svaku komponentu koju treba filtrirati
    for i in range(0, N_filters):
        C_current = C[i]
        
        # formiranje komplementa trenutnom centru komponente koja se filtrira
        C_complement = zeros_like(C_current)
        C_complement[0] = -C_current[0]
        C_complement[1] = -C_current[1]
        
        # formiranje osa centriranih oko komponente koja se filtrira
        if (M%2 ==0):
            y0 = y - C_current[0] + M/2 - 0.5 
        else:
            y0 = y - C_current[0] + (M-1)/2 
            
        if (N%2 ==0):
            x0 = x - C_current[1] + N/2 - 0.5 
        else:
            x0 = x - C_current[1] + (N-1)/2 
            
        X0, Y0 = meshgrid(x0, y0)
        
        # formiranje osa centriranih oko komponente koja se filtrira
        D0 = np.sqrt(np.square(X0) + np.square(Y0))
        
         # formiranje osa centriranih oko komplementa komponente koja se filtrira
        if (M%2 ==0):
            y0c = y - C_complement[0] - M/2 - 0.5 
        else:
            y0c = y - C_complement[0] - (M-1)/2 
            
        if (N%2 ==0):
            x0c = x - C_complement[1] - N/2 - 0.5 
        else:
            x0c = x - C_complement[1] - (N-1)/2 
            
        X0c, Y0c = meshgrid(x0c, y0c)
        
        # formiranje osa centriranih oko komponente koja se filtrira
        D0c = np.sqrt(np.square(X0c) + np.square(Y0c))
        
        # formiranje konacne maske sabiranjem svih prethodnih maski 
        # za sve prethodne komponente i trenutno izaracunatih
        filter_mask = filter_mask + \
            exp(-np.square(D0)/(2*np.square(radius))) + \
            exp(-np.square(D0c)/(2*np.square(radius)))
        
        # funkcija vraca inverznu filter masku, posto treba da potisne sve komponente iz niza C
        
    return 1-filter_mask
#####################################################################

# odredjivanje centara za notch filtre, bez komplementnih centara i bec DC vrednosti

x_diff = 155 # pomeraj po horizontalnoj osi
y_diff = 75 # pomeraj po vertikalnoj osi

x_iter_max = int(N/x_diff) # broj komponenti po horizontalnoj osi
y_iter_max = int(M/y_diff) # broj komponenti po vertikalnoj osi

C = [] # inicijalizacije liste za smestanje centara

y_iter =  0

# petlja za prikupljanje svih centara iz gornjeg levog dela slike podeljenje po dijagonali
# u C niz se dodaju iz centri sa donjeg levog dela dijagonale, ne racunajuci DC komponentu
for i in range(0,x_iter_max + 1):
    if (y_iter == y_iter_max + 1):
        y_iter = 0
        y_iter_max -= 1
    while y_iter < (y_iter_max + 1):
        if (y_iter == 8 and i == 8) or (y_iter == y_iter_max and i > x_iter_max/2):
            y_iter += 1
        else :
            C.append([ y_iter*75, i*155])
            y_iter += 1

#####################################################################

sigma = 25 # standardna devijacija filtra
H_filt = gaussian_notch_reject_filter( M, N, C, sigma) # notch-reject filtar

G = Fimg*H_filt # frekvencijsko filtriranje

# dobijena slika inverznom Furijeovom transformacijom filtriranog spektra 
g = real(ifft2(ifftshift(G)))
#  saturacija slike
g[g<0] = 0
g[g>1] = 1
#####################################################################

img_rescaled = exposure.rescale_intensity(g) # poboljsanje kontrasta dobijene slike

window_size = 13
sigma_color = 2
sigma_spatial = 5

# bilateralni filtar za odsumljavanje - sigma_color = 2, sigma_spatial = 5
# bilateralni filtar je upotrebljen i da bi se fino definisali prelazi, ivice
img_bilateral = denoise_bilateral(img_rescaled, window_size, sigma_color, sigma_spatial) 
#####################################################################

# prikaz rezultata dobijenih filtriranjem notch-reject Gausovim filtrima
fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize = (20, 20), dpi = 120)
ax = axes.ravel()

ax[0].imshow(Fimg_mag, cmap = 'gray')
ax[0].set_title('Amplitudski spektar originalne slike', fontsize = 16)
ax[0].set_axis_off()

ax[1].imshow(Fimg_ang, cmap = 'gray')
ax[1].set_title('Fazni spektar originalne slike', fontsize = 16)
ax[1].set_axis_off()

ax[2].imshow(log(1+abs(H_filt)), cmap = 'gray')
ax[2].set_title('Amplitudski spektar Gausovskog notch filtra, sigma = ' + str(sigma), fontsize = 16)
ax[2].set_axis_off()

ax[3].imshow(log(1+abs(G)), cmap = 'gray')
ax[3].set_title('Amplitudski spektar filtrirane slike' , fontsize = 16)
ax[3].set_axis_off()

ax[4].imshow(img, cmap = 'gray')
ax[4].set_title('Originalna slika', fontsize = 16)
ax[4].set_axis_off()

ax[5].imshow(g, cmap = 'gray')
ax[5].set_title('Dobijena filtrirana slika', fontsize = 16)
ax[5].set_axis_off()

ax[6].imshow(img_rescaled, cmap = 'gray')
ax[6].set_title('Skalirana filtrirana slika', fontsize = 16)
ax[6].set_axis_off()

ax[7].imshow(img_bilateral, cmap = 'gray')
ax[7].set_title('Bilateralno-filtrirana dobijena slika \n sigma_c = ' + str(sigma_color) + ', sigma_s = ' + str(sigma_spatial) + ', window = ' + str(window_size), fontsize = 16)
ax[7].set_axis_off()

plt.tight_layout()
plt.show()
#####################################################################

# finalno resenje
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 20), dpi = 120)
ax = axes.ravel()

ax[0].imshow(img, cmap = 'gray')
ax[0].set_title('Ulazna slika', fontsize = 16)
ax[0].set_axis_off()

ax[1].imshow(img_bilateral, cmap = 'gray')
ax[1].set_title('Izlazna slika', fontsize = 16)
ax[1].set_axis_off()
#####################################################################

# ucitavanje slike 
img_name = 'lena_noise.tif';
img = img_as_float(imread(folder_path + img_name))

# ocitavanje i ispisivanje dimenzija slike
[M, N] = shape(img)

print('Dimenzije slike su: \ndim = ' + str(M) + 'x' + str(N))

# prikaz ucitane slike
figure(figsize = (8, 8), dpi = 80);
imshow(img ,cmap = 'gray');
plt.title('Ulazna slika')
plt.axis('off')
plt.show()
#####################################################################

radius_num = 20 # broj razlicitih radijusa lokalnih susedstava
radius_arr = np.arange(1,radius_num + 1) # niz vrednosti za radijuse lokalnih susedstava
var_arr = [] # niz estimiranih varijansi za svaki vrednost radijusa  -
# iz histograma cemo naci najzastupljeniju varijansu za svako lokalno susedstvo


Nbins = 1000 # broj stubica kod histograma lokalnih varijansi

fig, axes = plt.subplots(nrows = 10, ncols = 2, figsize = (14, 12), dpi = 120)
ax = axes.ravel()

# petlja za svaki radius iz intervala 1-20
for i in range(radius_num):

    """
        Ideja je da se za svaku vrednost radijusa iz intervala [1,20]
        odredi lokalna varijansa slike. Ona se odredjuje tako sto se 
        prvo slika klasicno usrednji (img_avg), potom se kvadrirana slika usrednji (img_avg_sqr)
        i na kraju se po relaciji: img_var = img_avg_sqr - img_avg**2 dobije slika
        koja u sebi u svakom pikselu sadrzi samo lokalne varijanse za lokalno susedstvo
        definisano velicinom radijusa. 
        Nakon toga, iz tako dobijene slike, odnosno matrice lokalnih varijansi
        se odredjuje histogram te slike. Iz histograma dobijamo vrednosti na kojima 
        je neka varijansa najzastupljenija i koliko piksela ima tu varijansu.
        
    """
    
    # velicina prozora za usrednjavanje slike
    window_size = 2*radius_arr[i] + 1
    
    # formiranje normalizovanog prozora za lokalno usrednjavanje slike
    window =  np.ones((window_size, window_size))/(window_size**2)
    
    # usrednjavanje originalne slike
    img_avg = ndimage.correlate(img,window)

    # usrednjavanje kvadrata originalne slike
    img_avg_sqr = ndimage.correlate(img**2,window)
    
    # estimacija lokalne varijanse
    img_var = img_avg_sqr - img_avg**2
    
    # računanje histograma (histogram - y-osa, variance - x-osa)
    histogram, variance = np.histogram(img_var.flatten(), Nbins);
    
    # nalazenje najfrekventnije varijanse - max histograma, ali nam treba i index (to je varijansa zapravo - x-osa)          
    N_var_max = max(histogram.flatten())
    histogram = histogram.tolist()
    var_max_index = histogram.index(N_var_max)
    
    var_max = variance[var_max_index]
            
    # dodavanje izabrane varijanse
    var_arr.append([var_max, N_var_max])
    
    # iscrtavanje histograma
    ax[i].hist(img_var.flatten(), Nbins)
    # prikaz histogram za razlicite radijuse 
    # uz to se i ispisuje varijansa koja se najcesce pojavljuje na slici (var), i broj piksela koji imaju tu vrednost (N_max_var)
    ax[i].set_title('radius=' + str(radius_arr[i]) +  ', var=' + str(round(var_arr[i][0],5)) + ', N_max_var=' + str(var_arr[i][1]))

    
plt.tight_layout()
plt.show()
#####################################################################

var_arr = np.array(var_arr) # vracanje iz liste u array zbog plotovanja

# iscrtavanje dobijenih vrednosti varijansi i broj piksela sa tom varijansom
fig, ax1 = plt.subplots(figsize = (6, 4), dpi = 120)

color = 'tab:red'
ax1.set_xlabel('radijus [pixel]')
ax1.set_title('Zavisnosti varijanse i broja \npiksela sa tom varijansom od radijusa', fontsize = 10)
ax1.set_ylabel('varijansa', color=color)
ax1.plot(radius_arr, var_arr[:,0], '-or')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 

color = 'tab:blue'
ax2.set_ylabel('broj piksela', color=color)
ax2.plot(radius_arr, var_arr[:,1], '-ob')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.show()
#####################################################################

# prikaz dobijenih rezultata za radijuse od 4 do 10 (u tom intervalu su pojave estimirane varijanse najcesce, )
fig, axes = plt.subplots(nrows = 7, ncols = 3, figsize = (16, 30), dpi = 120)
ax = axes.ravel()
    
for i in range(4,11,1):
    radius = i
    var = var_arr[radius-1, 0] 

    # adaptivno filtriranje pomocu estimirane varijanse
    
    # velicina prozora za usrednjavanje slike
    window_size = 2*radius_arr[i] + 1
    
    # formiranje normalizovanog prozora za lokalno usrednjavanje slike
    window =  np.ones((window_size, window_size))/(window_size**2)
    
    # usrednjavanje originalne slike
    img_avg = ndimage.correlate(img,window)

    # usrednjavanje kvadrata originalne slike
    img_avg_sqr = ndimage.correlate(img**2,window)
    
    # estimacija lokalne varijanse
    img_var = img_avg_sqr - img_avg**2

    # formiranje adaptivne tezinske matrice
    w = var/img_var;
    w[w>1] =1

    # adaptivno filtiranja slika 
    img_est = img + w*(img_avg-img)
    
    j = i-4
    ax[3*j].imshow(img ,cmap = 'gray');
    ax[3*j].set_axis_off();
    ax[3*j].set_title('Ulazna slika');

    ax[3*j+1].imshow(w ,cmap = 'gray');
    ax[3*j+1].set_axis_off();
    ax[3*j+1].set_title('Adaptivna tezinska matrica \n radius=' + str(radius));

    ax[3*j+2].imshow(img_est ,cmap = 'gray');
    ax[3*j+2].set_axis_off();
    ax[3*j+2].set_title('Izlazna slika');
    
plt.tight_layout()
plt.show()
#####################################################################

# finalno resenje 
Nvar_max = max(var_arr[:,1])
radius = var_arr[:,1].tolist().index(Nvar_max) 

# adaptivno filtriranje pomocu estimirane varijanse
var = var_arr[radius-1, 0] 

print('Varijansa Gausovog suma: ' + str(var))

# velicina prozora za usrednjavanje slike
window_size = 2*radius_arr[i] + 1

# formiranje normalizovanog prozora za lokalno usrednjavanje slike
window =  np.ones((window_size, window_size))/(window_size**2)

# usrednjavanje originalne slike
img_avg = ndimage.correlate(img,window)

# usrednjavanje kvadrata originalne slike
img_avg_sqr = ndimage.correlate(img**2,window)

# estimacija lokalne varijanse
img_var = img_avg_sqr - img_avg**2

# formiranje adaptivne tezinske matrice
w = var/img_var;
w[w>1] =1

# adaptivno filtiranja slika 
img_est = img + w*(img_avg-img)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 10), dpi = 120)
ax = axes.ravel()

ax[0].imshow(img ,cmap = 'gray');
ax[0].set_axis_off();
ax[0].set_title('Ulazna slika');

ax[1].imshow(img_est ,cmap = 'gray');
ax[1].set_axis_off();
ax[1].set_title('Izlazna slika');
    
plt.tight_layout()
plt.show()
#####################################################################

# ucitavanje slike
img_name = 'road_blur.png';
img = img_as_float(imread(folder_path + img_name))

# ocitavanje dimenzija slike i ispis 
[M, N, D] = shape(img)

print('Dimenzije slike su: \ndim = ' + str(M) + 'x' + str(N) + 'x' + str(D)) 

# prikaz slike
figure(figsize = (10, 10), dpi = 80);
imshow(img, cmap = 'jet');
plt.title('Ulazna slika')
plt.axis('off')
plt.show()
#####################################################################

# Furijeova transformacija svih komponenti slike
F_r = fftshift(fft2(img[:,:,0]))
F_g = fftshift(fft2(img[:,:,1]))
F_b = fftshift(fft2(img[:,:,2]))

# iscrtavanje amplitudskih i faznih spektara svih komponenti slike
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 12), dpi = 120)
ax = axes.ravel()

ax[0].imshow(log(1+abs(F_r)) ,cmap = 'gray');
ax[0].set_axis_off();
ax[0].set_title('Amplitudski spektar ulazne slike \nR komponenta', fontsize = 16);

ax[1].imshow(log(1+abs(F_g)) ,cmap = 'gray');
ax[1].set_axis_off();
ax[1].set_title('Amplitudski spektar ulazne slike \nG komponenta', fontsize = 16);

ax[2].imshow(log(1+abs(F_b)) ,cmap = 'gray');
ax[2].set_axis_off();
ax[2].set_title('Amplitudski spektar ulazne slike \nB komponenta', fontsize = 16);

ax[3].imshow(np.angle(F_r) ,cmap = 'gray');
ax[3].set_axis_off();
ax[3].set_title('Fazni spektar ulazne slike \nR komponenta', fontsize = 16);

ax[4].imshow(np.angle(F_g) ,cmap = 'gray');
ax[4].set_axis_off();
ax[4].set_title('Fazni spektar ulazne slike \nG komponenta', fontsize = 16);

ax[5].imshow(np.angle(F_b) ,cmap = 'gray');
ax[5].set_axis_off();
ax[5].set_title('Fazni spektar ulazne slike \nB komponenta', fontsize = 16);

plt.tight_layout()
plt.show()
#####################################################################

P = 2*M-1
Q = 2*N-1
img_extended = zeros((P,Q,D))
img_extended[0:M, 0:N, :] = img


figure(figsize = (10, 10), dpi = 80);
imshow(img_extended, cmap = 'gray', vmin = 0, vmax = 1);
plt.title('Ulazna slika prosirena nulama')
plt.axis('off')
plt.show()
#####################################################################

# Furijeova transfomracija prosirene slike 
F_r_extended = fftshift(fft2(img_extended[:,:,0]))
F_g_extended = fftshift(fft2(img_extended[:,:,1]))
F_b_extended = fftshift(fft2(img_extended[:,:,2]))

# style.use('classic')
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 12), dpi = 120)
ax = axes.ravel()

ax[0].imshow(log(1+abs(F_r_extended)) ,cmap = 'gray');
ax[0].set_axis_off();
ax[0].set_title('Amplitudski spektar prosirene \nulazne slike - R komponenta', fontsize = 16);

ax[1].imshow(log(1+abs(F_g_extended)) ,cmap = 'gray');
ax[1].set_axis_off();
ax[1].set_title('Amplitudski spektar prosirene \nulazne slike - G komponenta', fontsize = 16);

ax[2].imshow(log(1+abs(F_b_extended)) ,cmap = 'gray');
ax[2].set_axis_off();
ax[2].set_title('Amplitudski spektar prosirene \nulazne slike - B komponenta', fontsize = 16);

ax[3].imshow(np.angle(F_r_extended) ,cmap = 'gray');
ax[3].set_axis_off();
ax[3].set_title('Fazni spektar prosirene slike \nR komponenta', fontsize = 16);

ax[4].imshow(np.angle(F_g_extended) ,cmap = 'gray');
ax[4].set_axis_off();
ax[4].set_title('Fazni spektar prosirene slike \nG komponenta', fontsize = 16);

ax[5].imshow(np.angle(F_b_extended) ,cmap = 'gray');
ax[5].set_axis_off();
ax[5].set_title('Fazni spektar ulaprosirenezne slike \nB komponenta', fontsize = 16);

plt.tight_layout()
plt.show()
#####################################################################

def restore_gaussian_filtered_image(img, sigma, wiener_k = 1e-4):
    
    """
        Opis: 
            Funkcija uzima ulaznu sliku, odredjuje njene dimenzije i 
            prosiruje nulama do duplih dimenzija.
            Racuna se frekvencijski lowpass Gausov filtar dimenzija kao prosirena
            slika za ulazni parametar sigma. 
            Za svaku komponentu ulazne slike se odredjuje spektar. Potom
            se racuna tezinska matrica Vinerovog filtra koji se koristi 
            pri procesu inverznog filtriranja, da bi se potisle lazne visoke frekvencije.
            Nakon filtriranja, dobija se slika restaurirana sa pretpostavljenim
            modelom degradacije.
        
        Parametri:
            img - ulazna slika za restauraciju
            sigma - standardna devijacija Gausovog filtra
            wiener_k - konstanta Vinerovog filtra
        
        Funkcija vraca restauriranu sliku, njen spektar i pretpostavljeni model degradacije slike 
    
    """
     
    # uzimanje dimenzija slike
    [M, N, D] = shape(img)
    
    D = 3 # setovanje dubine slike na 3 nivoa, ne uzimamo alfa komponentu iz RGBA
    
    # formiranje prosirene slike 
    P = 2*M-1
    Q = 2*N-1
    img_extended = zeros((P,Q,D))
    
    # prosirena slika u gornjem levom uglu sadrzi orignalnu sliku
    img_extended[0:M, 0:N, :] = img[:,:,0:3]
    
    # formiranje Gausovog lowpass filtra sa definisanom sigma
    H_gauss = gaussian_low_pass_filter(P, Q, sigma); # pretpostavljeni model degradacije 
    
    # inicijalizacija Furijeove transformacije za restauriranu sliku
    F_est = zeros((P, Q, D))
    
    # inicijalizacija restaurirane slike
    img_restored = zeros((M,N,D))
    
    # petlja po svakoj komponenti slike
    for i in range(D):
        # racunanje spektra prosirena ulazne slike po jednoj komponenti
        F = fftshift(fft2(img_extended[:,:,i]))
        
        # tezinska matrica Vinerovog filtra
        W = (abs(H_gauss)**2)/(abs(H_gauss)**2 + wiener_k)

        # inverzno filtriranje sa Vinerovom tezinskom matricom
        F_est[:,:,i] = (F/H_gauss)*W  
        # bez tezinske matrice bismo dobili velike vrednosti u spektru na visokim ucenstanostima
        # a matrica W ce to da anaulira - ovakva vrsta inverznog filtriranja sa tezinskom matricom
        # zapravo jeste konfiguracija Vinerovog filtra
        
        # vracanje dobijene slike iz frekvencijskog domena u prostorni
        img_est = real(ifft2(ifftshift(F_est[:,:,i])))
        
        # vracanje slike na stare dimenzije
        img_est = img_est[0:M, 0:N]
        
        # saturacija slike 
        img_est[img_est < 0] = 0
        img_est[img_est > 1] = 1
        
        # cuvanje komponente restaurirane slike
        img_restored[:,:,i] = img_est
    
    return img_restored, F_est, H_gauss

#####################################################################

# prikaz rezultata dobijenih procesom restauracije 
# sigma u opsegu [110, 120] je eksperimentalno dobijena kao najbolja
fig, axes = plt.subplots(nrows = 10, ncols = 3, figsize = (16, 32), dpi = 120)
ax = axes.ravel()

for i in range(10):
    sigma = 110 + i
    [img_restored, F, H] = restore_gaussian_filtered_image(img, sigma)
    
    ax[3*i].imshow(img_restored ,cmap = 'jet');
    ax[3*i].set_axis_off();
    ax[3*i].set_title('Sigma = ' + str(sigma));
    
    ax[3*i+1].imshow(log(1+abs(H)) ,cmap = 'gray');
    ax[3*i+1].set_axis_off();
    
    ax[1].set_title('Pretpostavljen model degradacije');

    ax[3*i+2].imshow(log(1+abs(F[:,:,0])) ,cmap = 'gray');
    ax[3*i+2].set_axis_off();
    
    ax[2].set_title('Amplitudski spektar nakon restuaracije \nR-komponenta');

plt.tight_layout()
plt.show()
#####################################################################

# za najbolju standardnu devijaciju je uzeta vrednosti 115
# za tu standardnu devijaciju se tablice kola najbolje vide
# generalno, sve vrednosti iz intervala [110, 120] su jako dobre 
best_sigma = 115

[img_restored_best, F_best, H_best] = restore_gaussian_filtered_image(img, best_sigma)

# prikaz restaurirane slike i dobijenih amplitudskih spektara po komponentama
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (36, 24), dpi = 120)
ax = axes.ravel()
    
ax[0].imshow(img_restored_best ,cmap = 'jet', vmin = 0, vmax = 1);
ax[0].set_axis_off();
ax[0].set_title('Restaurirana slika \nsigma = ' + str(best_sigma), fontsize = 30);

ax[1].imshow(log(1+abs(F_best[:,:,0])) ,cmap = 'gray');
ax[1].set_axis_off();
ax[1].set_title('AFK - R komponenta', fontsize = 30);

ax[2].imshow(log(1+abs(F_best[:,:,1])) ,cmap = 'gray');
ax[2].set_axis_off();
ax[2].set_title('AFK - G komponenta', fontsize = 30)

ax[3].imshow(log(1+abs(F_best[:,:,2])) ,cmap = 'gray');
ax[3].set_axis_off();
ax[3].set_title('AFK - B komponenta', fontsize = 30);

plt.tight_layout()
plt.show()

#####################################################################

# finalno resenje - restauirana slika
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 20), dpi = 120)
ax = axes.ravel()
    
ax[0].imshow(img, cmap = 'jet');
ax[0].set_axis_off();
ax[0].set_title('Ulazna slika', fontsize = 20);
    
ax[1].imshow(img_restored_best ,cmap = 'jet');
ax[1].set_axis_off();
ax[1].set_title('Izlazna slika', fontsize = 20);

plt.tight_layout()
plt.show()
#####################################################################

# zumiranje tablica 
tables = img_restored_best[200:280, 300:380]
old_tables = img[200:280, 300:380]

from skimage.color import rgb2lab, rgb2ycbcr, rgb2hsv, rgb2gray

# provlacenje izvucenih tablica kroz razlicite kolor sisteme
tables_lab = rgb2lab(tables)
tables_ycbcr = rgb2ycbcr(tables)
tables_hsv = rgb2hsv(tables)
tables_gray = rgb2gray(tables)

fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (16, 16), dpi = 120)
ax = axes.ravel()

ax[0].imshow(old_tables, cmap ='jet')
ax[0].set_title('Izdvojene tablice sa ulazne slike')
ax[0].set_axis_off()

ax[1].imshow(tables, cmap ='jet')
ax[1].set_title('Izdvojene tablice sa restaurirane slike')
ax[1].set_axis_off()

ax[2].imshow(tables_gray, cmap ='gray')
ax[2].set_title('Grayscale')
ax[2].set_axis_off()

ax[3].imshow(tables[:,:,0], cmap ='jet')
ax[3].set_title('RGB - R komponenta')
ax[3].set_axis_off()

ax[4].imshow(tables[:,:,1], cmap ='jet')
ax[4].set_title('RGB - G komponenta')
ax[4].set_axis_off()

ax[5].imshow(tables[:,:,2], cmap ='jet')
ax[5].set_title('RGB - B komponenta')
ax[5].set_axis_off()

ax[6].imshow(tables_lab[:,:,0], cmap ='jet')
ax[6].set_title('Lab - L komponenta')
ax[6].set_axis_off()

ax[7].imshow(tables_lab[:,:,1], cmap ='jet')
ax[7].set_title('Lab - a komponenta')
ax[7].set_axis_off()

ax[8].imshow(tables_lab[:,:,2], cmap ='jet')
ax[8].set_title('Lab - b komponenta')
ax[8].set_axis_off()

ax[9].imshow(tables_ycbcr[:,:,0], cmap ='jet')
ax[9].set_title('YCbCr - Y komponenta')
ax[9].set_axis_off()

ax[10].imshow(tables_ycbcr[:,:,1], cmap ='jet')
ax[10].set_title('YCbCr - Cb komponenta')
ax[10].set_axis_off()

ax[11].imshow(tables_ycbcr[:,:,2], cmap ='jet')
ax[11].set_title('YCbCr - Cr komponenta')
ax[11].set_axis_off()

ax[12].imshow(tables_hsv[:,:,0], cmap ='hsv')
ax[12].set_title('HSV - H komponenta')
ax[12].set_axis_off()

ax[13].imshow(tables_hsv[:,:,1], cmap ='jet')
ax[13].set_title('HSV - S komponenta')
ax[13].set_axis_off()

ax[14].imshow(tables_hsv[:,:,2], cmap ='jet')
ax[14].set_title('HSV - V komponenta')
ax[14].set_axis_off()
#####################################################################

# ucitavanje slike
img_name = 'lena.tif';
img = img_as_float(imread(img_name))

# uzimanje dimenzija slike i ispis
[M, N] = shape(img)

print('Dimenzije slike su: \ndim = ' + str(M) + 'x' + str(N)) 

# prikaz slike
figure(figsize = (10, 10), dpi = 80);
imshow(img, cmap = 'gray');
plt.title('Ulazna slika', fontsize = 16)
plt.axis('off')
plt.show()
#####################################################################

def filter_gauss(img, radius, sigma):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku img, odredjuje njene dimenzije, 
        prosiruje za duzinu radius sa svih strana i popunjava ivičnim 
        vrednostima, a potom nad njom vrsi filtriranje
        Gausovim filtrom u prostornom domenu. 
    Parametri:
        img - ulazna slika za filtriranje
        radius - radius Gausovog filtra
        sigma - standardna devijacija Gausovog filtra
    
        
    Funkcija vraca filtriranu sliku.
    
    """
    
    
    """
    Radi brzeg filtriranja, filtar je jednodimenzioni i 
    prilikom filtriranja se prolazi kroz 2 neugnjezdene for petlje 
    po distanci filtra. 
    Takav pristup ubrzava filtriranje, jer pomeramo sliku
    u odnosu na filtar, a ne filtar po svakom pixelu slike. 
    Dakle, mahom se koristi operator za grupisanje vrsta/kolona (:) 
    sto smanjuje kompleksnost problema.
    """
    
    if radius < 0 :
        print('Invalid parameter (radius)')
        return -1
    
    # DIMNENZIJE
    
    # odredjivanje dimenzija ulazne slike
    [M, N] = shape(img) 
    
    # PROSIRIVANJE SLIKE
    
    # prosirivanje slike sa radius-pixela sa svake strane
    img_pad = zeros((M+2*radius, N+2*radius))
    img_pad[radius:M+radius, radius:N+radius] = img
    
    # levi deo
    img_pad[radius:M+radius,0:radius] = img[:,0:1]
    
    # desni deo
    img_pad[radius:M+radius,N+radius:N+2*radius] = img[:,N-1:N]
    
    # gornji deo
    img_pad[0:radius,radius:N+radius] = img[0:1,:]
    
    # donji deo
    img_pad[M+radius:M+2*radius, radius:N+radius] = img[M-1:M,:]
    
    # gornji levi ugao
    img_pad[0:radius,0:radius] = img[0,0]
    
    # donji levi ugao
    img_pad[M+radius:M+2*radius,0:radius] = img[M-1,0]
    
    # gornji desni ugao
    img_pad[0:radius,N+radius:N+2*radius] = img[0,N-1]
    
    # donji desni ugao
    img_pad[M+radius:M+2*radius,N+radius:N+2*radius] = img[M-1,N-1]
    
    # FORMIRANJE FILTRA
    
    # formiranje jednodimenzionog Gausovog filtra 
    gauss = np.ones((2*radius+1))
    
    # vektor rastojanja
    distance = np.arange(1, radius+1)
    
    # koeficijenti jedne strane 1D Gaussovog filtra
    gauss[(radius+1):(2*radius+1)] = np.exp(-(distance**2)/(2*sigma**2))
    
    # kopiranje koeficijenata na simetricnu stranu Gausovog filtra
    gauss[0:(radius)] = np.flip(gauss[(radius+1):(2*radius+1)])
    
    # normalizacija koeficijenata Gausovog filtra, 
    # da se ne bi promenila srednja vrednost slike nakon filtriranja
    gauss /= sum(gauss)
    
    img_out = np.zeros_like(img_pad)
    
    # FILTRIRANJE 
    
    # mnozenje prosirene slike sa centralnom vrednoscu Gausovog filtra
    # ta vrednost treba svaki pixel da pogodi, pa je to na ovaj nacin 
    # brze odradjeno
    img_out = img_pad*gauss[radius]
    
    # filtrira se vertikalno po slici
    # filtriranje po svim vrstama i kolonama, pri cemu pomeramo sliku za odredjeno rastojanje iz
    # for petlje po radius-u 
    # u ovom slucaju Gausov 1D filtar je oblika matrica vrsta
    for i in range(radius):
        img_out[:, radius:N+radius] +=  img_pad[:, i:(N+i)]*gauss[i]
        img_out[:, radius:N+radius] +=  img_pad[:, (radius+i+1):(N+radius+i+1)]*gauss[radius+i+1]
       
    # dobijeni rezultat nam sad postaje stara slika, posto taj rezultat koristimo 
    # za dalje filtriranje 
    img_pad = img_out.copy()
    img_out *= gauss[radius] # mnozenje slike sa centrom Gausovog filtra zbog ponovnog filtriranja koje se nalazi u sledecoj for petlji
    
    # filtrira se horizontalno po slici
    # ovde se filtriraju sve vrste slike, ali kolone koje pripadaju samo originlanoj slici - posto sam
    # u prethodnoj for petlji vec prosao kroz prosirenu sliku filtrom, pa se ovako ne duplira 
    # ovde bi sustinski 1D Gausov filtar bio matrica kolona, ali posto for petljom kroz njega prolazim
    # onda i nije bitno
    for i in range(radius):
        img_out[radius:M+radius, radius:N+radius] += img_pad[i:(M+i), radius:(N+radius)]*gauss[i]
        img_out[radius:M+radius, radius:N+radius] += img_pad[(radius+1+i):(M+radius+1+i), radius:(N+radius)]*gauss[radius+1+i]
        
    # vracanje izlazne slike na dimenzije ulazne slike 
    img_out = img_out[radius:M+radius, radius:N+radius]
    
    return  img_out
#####################################################################

def filter_gauss_freq(img, radius, sigma):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku img, odredjuje njene dimenzije, 
        prosiruje za duzinu radius i popunjava ivičnim 
        vrednostima.
        Formira se Gausov filtar i prosiruje se na dimenziju 
        prosirene slike. Odredjuju se Furijeove transforamcije 
        prosirene slike i filtra. Mnozenjem dobijenih transformacija
        i potom odredjivanjem inverzne Furijeve transformacije tog proizvoda
        dobija se slika koja je filtrirana Gausovim filtrom, pri cemu
        je filtar napravljen u prostornom domenu, ali je slika filtrirana u
        frekvecijskom domenu.
        
    Parametri:
        img - ulazna slika za filtriranje
        radius - radius Gausovog filtra
        sigma - standardna devijacija Gausovog filtra
    
        
    Funkcija vraca filtriranu sliku.
    
    """
    
    if radius < 0 :
        print('Invalid parameter (radius)')
        return -1
    
    # DIMNENZIJE
    
    # odredjivanje dimenzija ulazne slike
    [M, N] = shape(img) 
    
    
    # PROSIRIVANJE SLIKE
    
    img_pad = np.zeros((M+radius, N+radius))
    img_pad[0:M,0:N] = img
    
    # POPUNJAVANJE PROSIRENE IVICNIM PIXELIMA
    
    # slika se popunjava ivicnim pikselima da 
    # ne bi doslo do crnih ivica pri filtriranju
    
    # desni deo
    img_pad[0:M,N:N+radius] = img[:,N-1:N]
    
    # donji deo
    img_pad[M:M+radius,0:N] = img[M-1:M,:]
    
    # donji desni ugao
    img_pad[M:M+radius,N:N+radius] = img[M-1,N-1]
    
    # Furijeva transformacija prosirene slike
    
    F = fftshift(fft2(img_pad))
    
    # FORMIRANJE FILTRA u prostonom domenu
    
    # formiranje 2D Gausovog filtra  
    gauss = np.ones((2*radius+1, 2*radius+1))
    coef = np.ones((2*radius+1))
    
     # vektor rastojanja
    distance = np.arange(1, radius+1)
    
    # 1D koeficijenti filtra, simetricni oko centra
    coef[(radius+1):(2*radius+1)] = np.exp(-(distance**2)/(2*sigma**2))
    coef[0:(radius)] = np.flip(coef[(radius+1):(2*radius+1)])
    coef = np.array(coef)[np.newaxis]
    
    # posto vazi simetricnost po osama za ovaj filtar 
    # (jer je eksponencijalnog tipa i po kvadratnom odstojanju od centra)
    # onda samo kvadriranjem vektora koeficijenata (matricno)
    # dobijamo celokupan filtar na laksi nacin
    gauss = coef.T*coef
    
    # normalizacija koeficijenata filtra, da bi ostala ista srednja vrednost slike
    gauss /= sum(gauss)
        
    # prosirivanje prostornog Gausovog filtra na dimenzije prosirene slike
    gauss_pad = np.zeros((M+radius, N+radius))
    gauss_pad[0:(2*radius+1), 0:(2*radius+1)] = gauss
    
    # FURIJEOVA TRANSFORMACIJA FILTRA 
    
    G = fftshift(fft2(gauss_pad))
    
    # filtriranje Gausovim filtrom
    # filtriranje u prostornom domenu predstavlja korelaciju (odnosno, konvoluciju slike i filtrom) 
    # a to se u frekvencijskom domenu preslikava u mnozenje ta dva spektra
    F_filt = F*G
    
    # inverzna Furijeova transformacija filtriranog spektra
    img_out = real(ifft2(ifftshift(F_filt)))
    
    # vracanje slike na dimenzije ulazne slike
    img_out = img_out[radius:(M+radius), radius:(N+radius)]
    
    return img_out    
#####################################################################

radius = np.array([5,15,25,35,45])
sigma = radius/3

fig, axes = plt.subplots(nrows = len(radius), ncols = 2, figsize = (20,38), dpi = 120)
ax = axes.ravel()
    
for i in range(len(radius)):
    
    img1 = filter_gauss(img, radius[i], sigma[i])
    img2 = filter_gauss_freq(img, radius[i], sigma[i])

    ax[2*i].imshow(img1, cmap = 'gray', vmin = 0, vmax = 1);
    ax[2*i].set_axis_off();
    ax[2*i].set_title('Prostorno filtriranje \nradius = ' + str(radius[i]), fontsize = 14);

    ax[2*i+1].imshow(img2 ,cmap = 'gray', vmin = 0, vmax = 1);
    ax[2*i+1].set_axis_off();
    ax[2*i+1].set_title('Frekvencijsko filtriranje \nradius = ' + str(radius[i]), fontsize = 14);

plt.tight_layout()
plt.show()
#####################################################################

from time import time
# memorija za cuvanje podataka o vremenu izvrsavanja funkcija 
# za razlicite radius-e
spatial_records = []
frequency_records = []

for i in range(1,51,1):
    radius = i
    sigma = radius/3
    
    # prostorno filtriranje 
    start_time = time()
    img_spatial = filter_gauss(img, radius, sigma)
    end_time = time()
    time_measured = end_time-start_time
    spatial_records.append(time_measured)
    
    # frekvencijsko filtriranje
    start_time_1 = time()
    img_frequency = filter_gauss_freq(img, radius, sigma)
    end_time_1 = time()
    time_measured_1 = end_time_1-start_time_1
    frequency_records.append(time_measured_1)
    
spatial_records = np.array(spatial_records)
frequency_records = np.array(frequency_records)
radius_arr = np.arange(1,51)
#####################################################################

# fitovanje dobijenih podataka na linearnu pravu za prikaz trenda
from scipy import optimize

def definition_for_function_to_fit(x, a, b):
    return a*x+b

params_1 = [0, 0] 
params_2 = [0, 0] 

params_1, covariance_1 = optimize.curve_fit(definition_for_function_to_fit, radius_arr, spatial_records, params_1)
params_2, covariance_2 = optimize.curve_fit(definition_for_function_to_fit, radius_arr, frequency_records, params_2)
#####################################################################

figure(figsize = (12, 8), dpi = 80);
plt.plot(radius_arr, spatial_records, 'xr')
plt.plot(radius_arr, frequency_records, 'ob')
plt.plot(radius_arr, definition_for_function_to_fit(radius_arr, *params_1), '-r')
plt.plot(radius_arr, definition_for_function_to_fit(radius_arr, *params_2), '-b')
plt.title('Zavisnost utroska vremena od radijusa', fontsize=12);
plt.xlabel('radius [pixel]', fontsize=10);
plt.ylabel('vreme [s]', fontsize=10);
plt.legend(['prostorno filtriranje', 'frekvencijsko filtriranje', 'prostorno filtriranje (fitovano)', 'frekvencijsko filtriranje (fitovano)'], loc = 'upper left')

plt.show()
#####################################################################

# Wavelet transformation with FFT

The wavelet transformation is a Fourier transformation (following here the numerical defintion, not the one used in physics) of a localized function:
$$
    F(t, \omega) = \int\limits_{-\infty}^\infty\! d\tau\, W(\frac{\tau-t}T) e^{-i\omega (\tau-t)} f(\tau),    
$$
where the $W$ window function satisfies
$$
   W(0)=1,\qquad W(\pm 1)=0, \qquad W(x) + W(1+x) = 1\quad\mathrm{for}\;x\in\{0,1\}. 
$$
For example
$$
    W(x) = \left\{\begin{array}{cc}
         1-f(x+1)\;&\mathrm{for}\; -1<x<-1/2  \\
         f(x)\;& \mathrm{for}\; -1/2<x<1/2 \\
         1-f(x-1)\;& \mathrm{for}\; 1/2<x<1
    \end{array}\right.
$$
Then
$$
    W(1+x) = \left\{\begin{array}{cc}
         1-f(x)\;&\mathrm{for}\; 0<x<1/2  \\
         f(x-1)\;& \mathrm{for}\; 1/2<x<3/2 \\
         1-f(x-2)\;& \mathrm{for}\; 3/2<x<2
    \end{array}\right.,
$$
and the sum rule is satisfied. For continuity $f(x)$ function then must satisfy $f(1/2)=1/2$, and for window-like function is has to take $f(0)=1$, but otherwise arbitrary. For example
$$
    f(x) = \frac 1{1+(2x)^p}
$$
with any $p$.

## Fourier transform

To get the Fourier transform of the original function, we can first make the wavelet transform, and do an additional discrete Fourier transformation. To see this we remark that the sum rule ensures that
$$
    \sum\limits_{n=-\infty}^\infty W(x+n) = 1.
$$
Thus
$$
    \sum\limits_{n=-\infty}^\infty e^{-i\omega(t+nT)} F(t+nT,\omega) =
    \int\limits_{-\infty}^\infty\! d\tau\, \sum\limits_{n=-\infty}^\infty W(\frac{\tau-t}T-n) e^{-i\omega \tau} f(\tau) 
    = {\cal F}(f)(\omega),
$$
where $\cal{F}$ denotes the Fourier transformation. This is true for all $t$, so we can safely choose $t=0$:
$$
    {\cal F}(f)(\omega) = \sum\limits_{n=-\infty}^\infty e^{-i\omega nT} F(nT,\omega).
$$

Alternatively, we can use multiply overlapping regions. We start from the formula
$$
    {\cal F}(f)(\omega) = \frac1{M} \sum_{m=0}^{M-1} \sum\limits_{n=-\infty}^\infty e^{-i\omega (t_m+nT)} F(t_m+nT,\omega).
$$
We can choose $t_m=mT/M$, then $t_m+Tn = (n+m/M)T$. We then may unite the two sums and write
$$
    {\cal F}(f)(\omega) = \frac1{M} \sum\limits_{n=-\infty}^\infty e^{-i\omega nT/M} F(nT/M,\omega).
$$

## Inverse transformation

The inverse Fourier transform reads
$$
    I(\tau,t) = \int\limits_{-\infty}^\infty\frac{d\omega}{2\pi} e^{i\omega\tau} F(t, \omega) = e^{i\omega t}W(\frac{\tau-t}T) f(\tau).
$$
This is the basic formula for synthetizing the original signal from the spectrogram:
$$
    f(\tau) = \sum_{n=-\infty}^\infty e^{-i\omega nT} I(\tau, nT).
$$
With overlapping regions it reads
$$
    f(\tau) = \frac 1M \sum_{n=-\infty}^\infty e^{-i\omega nT/M} I(\tau, nT/M).
$$


## Perform the integral with FFT

To calculate the integral we first shift it with $\tau\to\tau+t$, and note that the range is consized to $[-T,T]$:
$$
    F(t, \omega) = \!\! \int\limits_{-T}^T \!d\tau \, W(\frac{\tau}T) e^{-i\omega \tau} f(\tau+t).
$$
Usually numerically it is more convenient to work with solely positive integration/summation ranges. We should note that in this sense, the first sensible $t$ value is $T$, otherwise we get negative value in $f$. So, numerically, it is advantegous to shift the time with $T$, and introduce
$$
    \bar F(t, \omega) = F(t+T,\omega) = \!\! \int\limits_0^{2T} \!d\tau \, \bar W(\frac{\tau}T) e^{-i\omega (\tau-T)} f(\tau+t),
$$
where $\bar W(x)=W(x-1)$.  For consistency we will require $1= e^{-2i\omega\,T}$, meaning that $\omega$ is discrete, having values
$$ 
    \omega_k = k\frac\pi T.
$$
This also means $e^{i\omega_k T}= e^{ik\pi}=(-1)^k$. Therefore, using $F_k(t)=F(t,\omega_k)$, we obtain
$$
    \bar F_k(t) = (-1)^k \!\! \int\limits_0^{2T} \!d\tau \, \bar W(\frac{\tau}T) e^{-i\omega \tau} f(\tau+t).
$$
This formula will be discretized with $dt$ discretization steps, i.e. $\tau=m\,dt$, $t=n\,dt$. We introduce $N_T=T/dt$, and assume $N_T$ is integer. The discretized functions are $\bar W_m=\bar W(m\,dt/T) = W(m\,dt/T-1)$, $f_n = f(n\,dt)$ and $\bar F_n(\omega) = \bar F(n\,dt,\omega) = F((n+N_T)dt,\omega)$. We obtain
$$ 
    \bar F_{nk} = dt(-1)^k \sum\limits_{m=0}^{2N_T-1}  e^{-2\pi i\, mk /(2N_T)} \bar W_m f_{n+m}.
$$
This is exactly the discrete Fourier transform (DFT), up to a factor:
$$ 
    \bar F_{nk} = (-1)^k dt\,\mathrm{DFT}_k(\bar W f^{(n)}),
$$
where $f^{(\ell)}_n=f_{\ell+n}$. Therefore
$$
    F_{nk} = dt\,\mathrm{DFT}_k(\bar W f^{(n)}).
$$


## Inverse transformation

If we perform the inverse discrete Fourier transformation, we obtain expressions for all possible $n$
$$
    A^{(n)}_m = \frac1{dt}\, iDFT_m( F_{nk} ) = \bar W_m f_{n+m}.
$$
Since $\bar W_m = W(m\,dt/T-1)$ thus $\bar W_{m - k N_T/M} = W(m\,dt/T-k/M-1)$, and so
$$
    \frac1M \sum_{k=-\infty}^\infty \bar W_{m - k N_T/M} = 1.
$$
This implies
$$
    f_m = \frac1M \sum_k A^{(kN_T/M)}_{m-k N_T/M}.
$$
This is the inverse wavelet transformation.


## Harmonic function

The most simple example is the harmonic function $f=e^{i\omega_0 t}$. Its Fourier transform is Dirac-delta concentrated at $\omega=\omega_0$. The window function causes it a somewhat broader function
$$ \bar F = dt DFT(\bar W) $$
As the figure below suggests, for $T=0.1$ sec, the effective band width is about 20 Hz.

## Power

The height of the peak in a harmonic function is in connection with the full power in the original and in the Fourier transformed expresison.

We know that the total power of the original function in the interval $[-T,T]$ satisfies
$$
    P(t) = \int\limits_{-T}^T d\tau f^2(\tau+t) = \frac{dt^2}{2T}\sum_{k=0}^{2N_T-1} |\mathrm{DFT}(f)_k|^2.
$$

After creating windows we can calculate
$$    
    P'(t) = \int\limits_{-T}^T\! d\tau\, \left(W(\frac \tau T) f(\tau+t)\right)^2 = \frac{dt^2}{2T} \sum_{k=0}^{2N_T-1} |\mathrm{DFT}(Wf)_k|^2.
$$

If the frequency is much larger than the $1/T$, then we expect that is can be approximated as
$$
    P'(t) \approx \left\langle \left(W(\frac \tau T)\right)^2 \right\rangle \int\limits_{-T}^T\! d\tau\,  f^2(\tau+t) = {\cal N} P(t),
$$
where we introduced
$$
    {\cal N} = \left\langle \left(W(\frac \tau T)\right)^2 \right\rangle = \frac1{2T} \int\limits_{-T}^T d\tau\, W^2(\frac \tau T) = \frac12 \int\limits_{-1}^1 dx W^2(x).
$$
Thus
$$
    P(t) \approx \frac{dt^2}{2T\cal N}\sum_{k=0}^{2N_T-1} |\mathrm{DFT}(Wf)_k|^2.
$$
For real valued functions (that we have)
$$
    P(t) \approx \frac{dt^2}{T\cal N}\sum_{k=0}^{N_T-1} |\mathrm{DFT}(Wf)_k|^2.
$$
Using the spectrogram we find
$$
    P_n \approx \frac 1{T\cal N} \sum_{k=0}^{N_T-1} |F_{nk}|^2.
$$


## Logarithmic spectrum

At high frequencies the spactrum usually very messy, if the basic resonance frequencies are resided at low frequancies, say at 100-2000 Hz. To take into account this observation, we shall compute the logarithmic spectrum.

In general in the cumulative spectra the main objective is to maintain the energy relations, because the phases are expected to be random at large frequencies. Therefore we single out frequency windows with corner points $(\omega_0,\omega_1,\dots,\omega_n)$, and instead of the original frequencies, we take into account the power determined by the frequency intervals.
$$ 
    |F_{n\alpha}| =  \sqrt{\sum_{k=k_\alpha}^{k_{\alpha+1}} |F_{nk}|^2}.
$$
Clearly the total power is the same
$$
    P_n = \frac 1{T\cal N} \sum_\alpha |F_{n\alpha}|^2.
$$
For the phase we compute
$$
    \phi_{n\alpha} = \arg \sum_{k=k_\alpha}^{k_{\alpha+1}} F_{nk}.
$$
If there is a dominant frequency, $F$ will have a definite phase. If there are a lot of random contributors, then $F$ will have a random phase. We associate the phase of the frequency window $\alpha$ with the phase of $F_{n\alpha}$.

This way we generate the binned representation
$$
    F_{n\alpha} = |F_{n\alpha}| e^{\phi_{n\alpha}}.
$$


# FFT with set of resonator differential equations

## The damped harmonic oscillator

We seek the solution of the differential equation
$$
    \ddot x + 2\gamma\omega_0 \dot x + \omega_0^2 x = f(t),\qquad x(0)=x_0,\;\dot x(0) = v(0).
$$
In Fourier space
$$
    (-\omega^2 -2i\gamma\omega\omega_0 +\omega_0^2)x(\omega) = f(\omega).
$$
The roots of the left hand side are
$$
    \omega_\pm = \omega_0(-i\gamma\pm\sqrt{1-\gamma^2}) \quad\rightarrow\quad e^{-i\omega_\pm t} = e^{-\gamma\omega_0 t \pm i\Omega t},
$$
where $\Omega=\omega_0\sqrt{1-\gamma^2}$. So the general solution reads
$$
    x(\omega) = A_+\delta(\omega-\omega_+) + A_-\delta(\omega-\omega_-) + \frac{f(\omega)}{\omega_0^2-\omega^2-2i\gamma\omega\omega_0}.
$$
Because
$$
    \dot\Theta(t) =\delta(t) \quad\Rightarrow\quad -i\omega\Theta(\omega)=1  \quad\Rightarrow\quad \Theta(\omega)=\frac i\omega,
$$
moreover
$$
    f(\omega-\omega_0)  \quad\rightarrow\quad f(t) =\!\! \int\limits_{-\infty}^\infty\frac{d\omega}{2\pi}e^{-i\omega t}f(\omega-\omega_0) = e^{-i\omega_0 t} f(t),
$$
so it follows
$$
    {\cal FT}\left[\frac1{\omega-\omega_z}\right] = -i\Theta(t)e^{-i\omega_z t}
$$
The convolution property:
$$
    {\cal FT}[f(\omega)g(\omega)](t) = \int\limits_{-\infty}^\infty\!dt'\, f(t-t')g(t')
$$
and using
$$
    \frac{-f(\omega)}{(\omega-\omega_-)(\omega-\omega_+)} = \frac{f(\omega)}{2\Omega}\left(\frac1{\omega-\omega_-} - \frac1{\omega-\omega_+} \right),
$$
we find,
$$
    x(t) = A_-e^{-i\omega_-t} + A+-e^{-i\omega_+t} + \int\limits_{-\infty}^t\!dt' G(t-t') f(t'),
$$
where
$$
    G(t) = \frac1{\Omega} e^{-\gamma\omega_0 t} \sin(\Omega t)
$$
is the Green function of the problem. The particular solution can be simplified, using the fact that we can assume $f(t)=0$ for all $t<0$, so we can write
$$
    x(t) = A_-e^{-i\omega_-t} + A+-e^{-i\omega_+t} + \int\limits_0^t\!dt' G(t-t') f(t').
$$
Because the $x$ is real, so the initial conditions are real, too
$$
    x(t) = e^{-\omega_0 t}\left[ a\cos(\Omega t) + b\sin(\Omega t) \right]  + \int\limits_0^t\!dt' G(t-t') f(t').
$$
Now we can easily fit the initial conditions
$$
    x(t) = e^{-\omega_0 t}\left[ x_0 \cos(\Omega t) + \frac1{\Omega} (v_0 + \omega_0 x_0) \sin(\Omega t) \right]  + \int\limits_0^t\!dt' G(t-t') f(t').
$$
In particular, if $x_0=0$, $v_0=0$ and $f(t)=\delta(t)$, we find
$$
    x(t) = e^{-\omega_0 t}\frac1{\Omega}\sin(\Omega t).
$$
This shows that a change in $f(t)$ is followed by the solution with $t\sim1/\omega_0$ latency.

If $f(t)=\cos(\bar \omega t)$, i.e. $f(\omega) = \frac12(\delta(\omega-\bar\omega)+\delta(\omega-\bar\omega))$, thus
$$
    x(t) = e^{-\omega_0 t}\left[ a\cos(\Omega t) + b \sin(\Omega t) \right]  + \Theta(t) \frac12\left[\frac{e^{-i\bar \omega t}}{\omega_0^2-{\bar\omega}^2-2i\gamma\omega_0{\bar \omega}} + \frac{e^{i\bar \omega t}}{\omega_0^2-{\bar\omega}^2+2i\gamma\omega_0{\bar \omega}}\right],
$$
which simplifies to
$$
    x(t) = e^{-\omega_0 t}\left[a\cos(\Omega t) + b\sin(\Omega t) \right]  + \Theta(t) \frac{\cos(\bar\omega t)(\omega_0^2-{\bar\omega}^2) + 2\sin(\bar\omega t)\gamma\omega_0\bar\omega}
        {(\omega_0^2-{\bar\omega}^2)^2 + 4\gamma^2\omega_0^2{\bar \omega}^2} .
$$
Here we can fit any initial conditions, which is skipped here. The lesson is that the $t\to\infty$ solution is approached after a $\sim1/\gamma$ time delay, as it was the case for the Dirac-delta force. The other lesson is that the amplitude of the asimptotic solution is
$$
    A(\bar\omega, \omega_0,\gamma) = \frac1{\sqrt{(\omega_0^2-{\bar\omega}^2)^2 + 4\gamma^2\omega_0^2{\bar \omega}^2}}.
$$
The maximal value is reached when
$$
({\bar\omega}^2-\omega_0^2)^2 + 4\gamma^2\omega_0^2 {\bar\omega}^2 = \mathrm{minimal},
$$
thus
$$
    \bar\omega = \omega_0\sqrt{1 - 2\gamma^2}.
$$

The physical meaning of the $\gamma$ is the relative damping. $1/e$ portion of the initial conditions disappear when $\gamma\omega_0 t_e = 1$. This means in time that
$$
    t_e = \frac{T_{period}}{2\pi\gamma}
$$

## Recursion

We can have an analogous recursion problem. Let us assume the the data are given in $dt$ timesteps, and we have the data as
$$
f_n = f(n\,dt).
$$
Now we write 
$$
    \begin{aligned}
        x_{n+1} &= x_n + dt\,\omega_0 p_n \\
        p_{n+1} &= p_n + dt\,(f_n - 2\gamma p_n - \omega_0 x_n).  \\
    \end{aligned}
$$
In matrix form
$$
    X_{n+1} = \begin{pmatrix}0\cr dt f_n\cr\end{pmatrix} + 
    \begin{pmatrix}1 & dt \omega_0\cr -dt\omega_0 & 1-2dt\gamma\cr \end{pmatrix}X_n
$$
Its solution reads
$$
    X_ n = e^{nQ}X_0 + \sum_{m=0}^{n-1} e^{(n-m)Q} F_m
$$

We shall emphasize that this recursion is not the solution of the differential equation, especially for finite $dt$. Indeed, the exact solution for the recursion can be searched in the form


---
layout: post
title: "Cross-Links Part 2: The Gaussian Approximation"
date: 2020-01-20
---

<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

\newcommand{\n}{\boldsymbol{n}}
\newcommand{\no}{\boldsymbol{n_0}}
\newcommand{\strain}{\boldsymbol{\underline{\underline{\lambda}}}}
\newcommand{\xlink}{\boldsymbol{\underline{\underline{\ell}}}}
\newcommand{\lo}{\boldsymbol{\underline{\underline{\ell}}}_0}

This post is part 1 of a series where I'll be exploring the physics and mathematics of molecular cross-links in polymer networks.

Sections to write:
-What are cross-links?
-Examples?
-What features do we want to model? (prob dists, entropy, free energy)





Isotropic cross-linking
We model each cross-link as a polymer chain of length $L$ and end-to-end distance $\boldsymbol{R}$. As such, the distribution of $\boldsymbol{R}$ is a Gaussian of the form
\begin{equation}\label{isotropicdistribution}
P(\boldsymbol{R}) = \left[\frac{3}{2\pi R_0^2}\right]^{3/2} \exp \left(\frac{-3}{2R_0^2} \boldsymbol{R}\cdot\boldsymbol{R}\right).
\end{equation}
Here $R_0^2 = \ell_0L$, where $\ell_0$ is the effective step length of the polymer chain.

The free energy for a single cross-link with end-to-end distance $\boldsymbol{R}$ is entirely entropic, and is thus given by 
\begin{equation}\label{isofe}
\begin{aligned}
\mathscr{F}(\boldsymbol{R})  & = -k_BT\ln P(\boldsymbol{R}) \\
& =  \frac{3k_BT}{2R_0^2}\boldsymbol{R}\cdot\boldsymbol{R} -\frac{3k_BT}{2} \ln\left(\frac{3}{2\pi R_0^2}\right).
\end{aligned}
\end{equation}

We average this free energy over the probability distribution $P(\boldsymbol{R})$ of all cross-link orientations, which gives us 
\begin{equation}
\begin{aligned}
\langle \mathscr{F}\rangle & = \frac{3k_BT}{2R_0^2} \langle \boldsymbol{R}\cdot\boldsymbol{R}\rangle -\frac{3k_BT}{2} \ln\left(\frac{3}{2\pi R_0^2}\right)\\
&= \frac{3k_BT}{2R_0^2} \left(R_0^2\right) -\frac{3k_BT}{2} \ln\left(\frac{3}{2\pi R_0^2}\right)\\
&= \frac{3}{2}k_BT \left[ 1+\ln\left(\frac{2\pi R_0^2}{3}\right)\right]
\end{aligned}
\end{equation}
for the free energy of the ensemble of configurations of a single cross-link.
We find that we our cross-link free energy is almost equipartition, with a correction factor dependent on the mean cross-link end-to-end distance.

We now consider the effect of a strain on the cross-linked collagen fibril, which we quantify using the strain tensor, $\strain$. With regard to the cross-linking, the key effect of the strain is to alter the  end-to-end cross-link distance. The strain field causes the transformation $\boldsymbol{R}_\text{new} = \strain \boldsymbol{R}$. The new free energy per cross-link $\mathscr{F}(\boldsymbol{R}_\text{new})$ is of similar form to Eq. ~\ref{isotropicdistribution}.

To calculate the new free energy per cross-link, we plug $\boldsymbol{R}_\text{new}$ into Eq.~\ref{isofe} and average over the pre-strain cross-linking distribution. The resulting free energy is

\begin{equation}\label{isotropicfedensity}
\begin{aligned}
\langle\mathscr{F}\rangle & =  \frac{3k_BT}{2R_0^2} \langle\boldsymbol{R}_\text{new}^\top\cdot\boldsymbol{R}_\text{new}\rangle_{P(\boldsymbol{R})}\\
& = \frac{3k_BT}{2R_0^2} \langle\boldsymbol{R}^\top\cdot\strain^\top\cdot\strain\cdot\boldsymbol{R}\rangle_{P(\boldsymbol{R})} \\
& = \frac{3k_BT}{2R_0^2} \left\langle\text{Tr}\left[\boldsymbol{R}^\top\cdot\strain^\top\cdot\strain\cdot\boldsymbol{R}\right]\right\rangle_{P(\boldsymbol{R})} \\
& = \frac{3k_BT}{2R_0^2} \left\langle\text{Tr}\left[\strain^\top\cdot\strain\cdot\boldsymbol{R}\cdot\boldsymbol{R}^\top\right]\right\rangle_{P(\boldsymbol{R})}\\
& = \frac{3k_BT}{2R_0^2} \text{Tr}\left[\strain^\top\cdot\strain\cdot\left\langle\boldsymbol{R}\cdot\boldsymbol{R}^\top\right\rangle_{P(\boldsymbol{R})}\right] \\
& = \frac{3k_BT}{2R_0^2} \text{Tr}\left[\strain^\top\cdot\strain\cdot \left(\frac{R_0^2}{3}\underline{\underline{\delta}}\right)\right]\\
& = \frac{1}{2}k_BT \text{Tr}\left[\strain^\top\cdot\strain\right].
\end{aligned}
\end{equation}
Here we have omitted an additive constant, which we shall continue to do going forward.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Anisotropic cross-linking}

To model anisotropic cross-linking, we replace the scalar effective step length $\ell_0$ with the tensor \begin{equation}
\boldsymbol{\underline{\underline{\ell}}}_0 = \ell_\perp \boldsymbol{\underline{\underline{\delta}}} + [\ell_\parallel - \ell_\perp]\boldsymbol{n_0}\otimes \boldsymbol{n_0},
\end{equation}
which describes the anisotropy of the cross-link network. Here $\ell_\parallel$ is the effective step length parallel to the local director field $\no$, while $\ell_\perp$ is the perpendicular effective step length. We also define the dimensionless ratio $\zeta=\ell_\parallel/\ell_\perp$, which we will later make use of in order to simplify expressions. We then have the end-to-end cross-link distance distribution

\begin{equation}\label{anisprob}
P(\boldsymbol{R}) = \left[ \left(\frac{3}{2\pi L}\right)^3 \frac{1}{\textnormal{Det}[\lo]}\right]^{1/2} \exp{\left(-\frac{3}{2L}\boldsymbol{R^\top\lo^{-1}R}\right)}.
\end{equation}
Note here that $\lo^{-1} = 1/\ell_\perp \boldsymbol{\underline{\underline{\delta}}} + [1/\ell_\parallel - 1/\ell_\perp]\boldsymbol{n_0}\otimes \boldsymbol{n_0}$. We show that this is the correct form for the inverse of $\lo$ in the Appendix.


\begin{comment}
The free energy for a single cross-link is thus:
\begin{equation}\label{anisfe}
\begin{aligned}
\mathscr{F}(\boldsymbol{R}) & = \frac{3k_BT}{2L} (\boldsymbol{R^\top\lo^{-1}R}).
\end{aligned}
\end{equation}

We can then calculate the average free energy per crosslink \cite{warner1996nematic}:
\begin{equation}
\begin{aligned}
\langle\mathscr{F}\rangle & = \frac{3k_BT}{2L} \langle\boldsymbol{R^\top\lo^{-1}R}\rangle_{P(\boldsymbol{R})} \\
& = \frac{3k_BT}{2L} \langle\boldsymbol{R^\top\lo^{-1}R}\rangle_{P(\boldsymbol{R})} \\
& = \frac{3k_BT}{2L} \text{Tr}\left[ \lo^{-1} \cdot \left\langle\boldsymbol{R}\cdot\boldsymbol{R}^\top\right\rangle_{P(\boldsymbol{R})} \right] \\
& = \frac{3k_BT}{2L} \text{Tr}\left[ \lo^{-1} \cdot \left(\frac{L}{3}\lo\right) \right]\\
& = \frac{1}{2}k_BT\textnormal{Tr}\left[ \lo \lo^{-1}\right] \\
&= \frac{3}{2}k_BT,
\end{aligned}
\end{equation}
where we again recover equipartition.
\end{comment}

Now, we again consider the effect of a strain field $\strain$ acting on the fibril. The free energy per cross-link $\mathscr{F}(\boldsymbol{R}_\text{new})$ (where $\boldsymbol{R}_\text{new} = \strain \boldsymbol{R}$) is analogous to Eq. ~\ref{isotropicfedensity}. Thus, as before, we average $\mathscr{F}(\boldsymbol{R}_\text{new})$ with respect to the probability distribution $P(\boldsymbol{R})$ from Eq. ~\ref{anisprob}. This gives us
\begin{equation}
\begin{aligned}
\langle \mathscr{F}\rangle & = \frac{3k_BT}{2L} \langle\boldsymbol{R}_\text{new}^\top\cdot\xlink^{-1}\cdot\boldsymbol{R}_\text{new}\rangle_{P(\boldsymbol{R})} \\
& = \frac{3k_BT}{2L} \langle\boldsymbol{R}^\top\cdot\strain^\top\cdot\xlink^{-1}\cdot\strain\cdot\boldsymbol{R}\rangle_{P(\boldsymbol{R})}\\
& = \frac{3k_BT}{2L} \text{Tr} \left[ \strain^\top\cdot\xlink^{-1}\cdot\strain\cdot\langle\boldsymbol{R}^\top\cdot\boldsymbol{R}\rangle_{P(\boldsymbol{R})}\right] \\
& = \frac{3k_BT}{2L} \text{Tr} \left[ \strain^\top\cdot\xlink^{-1}\cdot\strain\cdot\left( \frac{L}{3}\lo\right)\right]\\
& = \frac{1}{2}  k_BT\textnormal{ Tr} [\lo\cdot \strain^\top\cdot \xlink^{-1}\cdot \strain].
\end{aligned}
\end{equation}

From here, we can get a free energy density due to the cross-linking by multiplying the average free energy per cross-link ($\langle\mathscr{F}\rangle$) by the cross-link density within the fibril, $\rho$. We then have the standard anisotropic elastomeric contribution\cite{warnertextbook}: 
\begin{equation}\label{maineq}
\boxed{f_\textnormal{Cross-Link} = \frac{1}{2} \rho k_BT\textnormal{ Tr} (\lo \strain^\top \xlink^{-1} \strain).}
\end{equation}

We define $\mu=\rho k_BT$, which is the elastic shear modulus for the fibril \cite{warner1996nematic}. This parameter has been found to have a wide range of values experimentally, but observations generally fall in the range of $10^6 - 10^8$ Pa for \textit{in vivo} collagen fibrils \cite{dutov2016measurement}\cite{quigley2018combining}.

The free energy per unit volume is then
\begin{equation}\label{xlinkfe}
E_\text{Cross-Link} = \frac{2}{R^{2}}\int_0^{R} r f_\text{Cross-Link} dr = \frac{\mu}{R^2}\int_0^{R} r\textnormal{ Tr} (\lo \strain^\top \xlink^{-1} \strain)dr ,
\end{equation}
where $R=R_0/\sqrt{\epsilon}$ is the post-strain fibril radius, and $R_0$ is the pre-strain equilibrium radius.

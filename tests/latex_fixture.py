"""Small LaTeX corpus generated on the fly for parser/pipeline tests."""

from __future__ import annotations

from pathlib import Path

MASTER = r"""
\documentclass{beamer}
\newcommand{\bX}{\boldsymbol{X}}
\newcommand{\bt}[1]{\textbf{#1}}
\newtheorem{definition}{Definition}
\begin{document}
\begin{frame}{Références}
  \includegraphics[width=5cm]{figures/davis}
\end{frame}
% CHAPTER [image path]{rgb defined color}{title}
\begin{chapter}[images/a.jpg]{leoyel}{Chapter 1}
	\textit{Les Processus ARMA}
\end{chapter}
\include{UTS_C1}
\begin{frame}{References}
  \begin{thebibliography}{9}
  \bibitem[A79]{A79} Ansley (1979).
  \end{thebibliography}
\end{frame}
\end{document}
"""

CHAPTER = r"""
\section[Intro]{Introduction}
\begin{frame}{Les séries temporelles}
\begin{definition}[Processus]
  Un \textbf{processus stochastique} noté $\{X_{t}\}_{t \in \mathbb{Z}}$ est une séquence
  de variables aléatoires. % comment
\end{definition}
\begin{definition}
  Une \textbf{série temporelle} est un segment des réalisations d'un processus.
\end{definition}
\begin{itemize}
  \item[Note 1] on suppose que le temps est discret
  \medskip
  \item Le vecteur $\bX$ est en gras \bt{important} 50\%
\end{itemize}
\note{le symbole $\supseteq$ dénote un sur-ensemble}
\end{frame}
%\begin{frame}{Ergodicité : l'intuition}
%  \begin{itemize}\item Commenté\end{itemize}
%\end{frame}
\begin{frame}
\frametitle{Stationnarité}
\begin{columns}
\begin{column}{0.5\textwidth}
  Un processus est \alert{stationnaire au second ordre} si
  \begin{align}
    E(X_t) &= \mu \\
    Cov(X_t, X_{t-h}) &= \gamma(h) \label{eq:stat}
  \end{align}
\end{column}
\begin{column}{0.5\textwidth}
  \begin{figure}
    \includegraphics{fig}
    \caption{Une trajectoire}
  \end{figure}
\end{column}
\end{columns}
\end{frame}
\section{Tableaux}
\begin{frame}[fragile]{Un tableau}
\begin{table}
\begin{tabular}{lcc}
\toprule
Modèle & AIC & BIC \\
\midrule
AR(1) & 10.2 & 11.0 \\
MA(1) & 9.8 & 10.9 \\
\bottomrule
\end{tabular}
\end{table}
\pause
\begin{lstlisting}
import numpy as np
\end{lstlisting}
\end{frame}
"""

MANUSCRIPT = r"""
\documentclass{book}
\begin{document}
\chapter{Modélisation}\label{ch:model}
Ce chapitre présente la modélisation des processus non-causaux. Il s'agit d'un paragraphe
suffisamment long pour être conservé dans le corpus de test.

\section{La filtration}\label{sec:filter}
Premier paragraphe de la section sur la filtration, avec une équation
\begin{equation}
  y_t = \sum_{k} d_k \varepsilon_{t-k}.
\end{equation}

Deuxième paragraphe de la section, qui parle d'autre chose et qui est également assez long
pour constituer un contenu retrouvable.

\begin{theorem}[Représentabilité]
Soit $(X_t)$ un processus stable. Alors il est représentable si et seulement si la suite $(d_k)$ est causale.
\end{theorem}

Un dernier paragraphe pour conclure la section et vérifier le regroupement des paragraphes.
\end{document}
"""

COMMON = r"""
\documentclass{beamer}
\begin{document}
\begin{frame}{Rappels d'algèbre}
Une matrice $A$ est inversible si $\det(A) \neq 0$. Rappel utile pour tous les niveaux.
\end{frame}
\end{document}
"""


def make_course_tree(root: Path) -> Path:
    """Create ``root/latex/{M1,M2,Commun}/...`` and return the latex directory."""
    latex = root / "latex"
    (latex / "M1" / "Univariate_Time_Series").mkdir(parents=True)
    (latex / "M1" / "Univariate_Time_Series" / "UTS.tex").write_text(MASTER, encoding="utf-8")
    (latex / "M1" / "Univariate_Time_Series" / "UTS_C1.tex").write_text(CHAPTER, encoding="utf-8")
    (latex / "M2" / "Research_Econometrics").mkdir(parents=True)
    (latex / "M2" / "Research_Econometrics" / "hdr_main.tex").write_text(MANUSCRIPT, encoding="utf-8")
    (latex / "Commun").mkdir()
    (latex / "Commun" / "algebra.tex").write_text(COMMON, encoding="utf-8")
    return latex

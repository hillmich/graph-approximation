from string import Template

tikz_main = Template(r'''
\documentclass[crop,tikz]{standalone}%
\begin{document}
\begin{tikzpicture}[every node/.style={ultra thick,circle,inner sep=0pt}, line width=1.5pt]
${background}
${graph}
\end{tikzpicture}
\end{document}
''')

tikz_node = Template(r'    \node[minimum size=${r}pt, draw=${color}] (${node_id}) at (${x}pt, -${y}pt) {};')
tikz_edge = Template(r'    \draw (${node_a}.center) -- (${node_b}.center);')
tikz_background = Template(r'    \node[anchor=north west, rectangle, draw=none] (bg) at (-5pt, 3pt) {\includegraphics{${path}}};')


transition_main = Template(r'''
\documentclass[crop,tikz]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
\begin{tikzpicture}[degree/.style={circle, minimum size=1cm, very thick, draw=#1},
                    transition/.style={-latex, thick, color=#1}]
    \pgfmathsetmacro{\polygonAngle}{360/${num_nodes}}
    \def\radius{6}
    \node[degree=black] (dX) {$$\varnothing$$};
${nodes}       
${edges}
\end{tikzpicture}
\end{document}
''')
transition_node = Template(r'    \node[degree=${color}] (d${degree}) at ({sin(\polygonAngle*${degree})*\radius}, {cos(\polygonAngle*${degree})*\radius}) {$$${label}$$};')
transition_edge = Template(r'    \draw[transition=${color}] (d${source}) edge[bend left=${bend}] node[sloped, above] {$$${prob}$$ $$(${lifetime})$$} (d${target});')

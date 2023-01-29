# NM-proj

Projekt na Numerické metody pro inženýry (P413003)  
Vypracoval Jiří Zbytovský ([email](zbytovsi@vscht.cz))

## Instalace & spuštění

Program byl vyvíjen v pythonu 3.11, v prostředí pipenv, v konzoli Windows cmd.

Kromě toho jsou zde použity knihovny `numpy` (na řešení soustav lin. rovnic), `scipy` (na řešení ODE), `matplotlib` (vykreslení grafu) a `click` (rozhraní). Instalace doporučena pomocí `pipenv`, avšak lze spustit i ve výchozím prostředí pomocí `pip`.  
Do konzole tedy zadejte následující příkazy jedním z těchto postupů.  
Výsledkem je graf řešení či výstup do konzole _(pozn. pro pokračování je třeba zavřít graf)_

### A) pipenv

```bat
pipenv install
pipenv run app\uloha_1.py
pipenv run app\uloha_2.py
```

### B) výchozí python na Windows

```bat
py -3 -m venv venv
venv\scripts\activate
python -m pip install -r requirements.txt
python app\uloha_1.py
python app\uloha_2.py
deactivate
```

### Parametry

Výše popsaný postup spustí obě úlohy s vhodnými výchozími paramatery.  
Parametr `--help` zobrazí možnosti argumentů.

V úloze 1 lze zadat:
- `eps` = _ε_ (mez konvergence _θ_)
- `ode-step` = maximální krok _x_ při řešení ODE, určuje přesnost tohoto řešení

V úloze 2 lze zadat:
- `n` = počet úseků, na který je rozdělen interval _x_
- `m` = počet úseků, na který je rozdělen interval _t_
- `delta` = bezrozměrná rychlostní konstanta reakce _(jen pro experimentování, jinak 1 jako zadaná hodnota)_
- `diff` = bezrozměrný difúzní koeficient _(jen pro experimentování, jinak 1 jako zadaná hodnota)_

Např. `pipenv run uloha_2.py --delta 2e-4 --diff .3` poskytuje esteticky zajímavý výsledek...

## Development

`pipenv install --dev`

Run prettier:
`pipenv run prettier`

Run lint:
`pipenv run lint`

Compile LaTeX (cd tex):
`pdflatex protokol.tex`

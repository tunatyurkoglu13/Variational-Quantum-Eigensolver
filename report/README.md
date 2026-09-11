# Building the report

Requires a LaTeX distribution with `biber` (biblatex's backend). With BasicTeX:

```
brew install --cask basictex
# open a new terminal so tlmgr is on PATH, then:
sudo tlmgr update --self
sudo tlmgr install physics siunitx biblatex biber logreq
```

Then compile:

```
cd report
latexmk -pdf main.tex
```

CI builds this automatically on every push (see `.github/workflows/report.yml`)
using a containerized LaTeX action, so a reviewer never needs local LaTeX to
get the current PDF -- it's a build artifact on every CI run.

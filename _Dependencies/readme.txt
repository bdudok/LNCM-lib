the _Dependencies folder should have following modules installed, and updated by git.
Add subfolders to .gitignore
.gitignore: /_Dependencies/PyControl/

1) PyControl
mkdir LNCM-lib/_Dependencies/PyControl
cd LNCM-lib/_Dependencies/PyControl
git clone https://github.com/pyControl/code.git

as of 1/23/2024, most recent PyControl version switched to new data file format and
reader is not backwards compatible with older files. Pushing the old version to the repo until we update recorder.
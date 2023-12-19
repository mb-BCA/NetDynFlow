# TO-DO list before release of v2.

#### Before merging new version into master:

- Open a new branch out of dev_v2 --> '**v2-cleanup**'.
- Clean ____init____.py: 
	- Remove imports to test modules. 
	- Double check list of absolute imports.
	- Update general description. Explanation for the canonical models.
	- Double check the instructions in the initial docstring. New names of functions. Make sure the examples work if copy/pasted.
	- Update the publication list, include the latest dynflow paper.
- Update the Copyright dates in all files.
- Update the version number to 2.0.0 in ____init___.py and setup.py.
- Remove unnecessary files (to-do lists, NamingConventions, etc.)
- Update the **README.md** file.
	- Remove initial unnecessary text. 
	- Update general description(s), include explanation for the different canonical models.
	- Update the Copyright/license infos.
	- Double check the instructions in the initial docstring. New names of functions. Make sure the examples work if copy/pasted.
	- Update list of changes.

- Finally: merge the thest branch. 

#### After the merge:

- Give all the steps to make the repository / package installable.
- Verify installation works.
- (If needed) Update the installation instructions in README.md + ____init____.py.
- Create a (GitHub) release.
- Add the library to PYPI (?)



## PyCharm workspace setup

PyCharm saves a project's settings in a `.idea/` folder at the root of whatever folder you open. That includes the interpreter (`.idea/misc.xml`), run configurations, and window layout. If you open a repo folder directly, `.idea/` ends up inside the repo, and every PyCharm project opened on that repo shares it. That causes two problems:

- Projects overwrite each other's interpreter and settings.
- IDE files end up as untracked or committed clutter in git.

To avoid this, **keep PyCharm project folders outside the repos** and add the repos to them as content roots.

### Setup

1. **If you already worked in a repo with Pycharm, clean the repo.** Close PyCharm. Delete the `.idea/` folder inside the repo, or move it somewhere as a backup. Then stop git from picking it up again. Using Git Bash:

   ```bash
   # ignore locally (not committed)
   echo ".idea/" >> .git/info/exclude

   # if .idea/ was already committed
   git rm -r --cached .idea
   git commit -m "Stop tracking PyCharm project files"
   ```

2. **Create a wrapper folder per setup**, outside any repo:

   ```
   ~/PycharmProjects/2p-analysis/
   ~/PycharmProjects/lncm-dev/
   ```

3. **Open the wrapper in PyCharm**, then go to
   *Settings → Project Structure → Add Content Root* and choose the repo.
   - If the wrapper folder itself is listed as a content root, remove it.
   - One wrapper can hold several repos (e.g. `LNCM-lib` + `2p_code`), so navigation works across them.
   - Check that *Settings → Version Control → Directory Mappings* lists each repo.

4. **Set the interpreter per project**:
   *Settings → Project → Python Interpreter*.

Each wrapper now has its own `.idea/` with its own interpreter, run configurations and layout, while all of them point at the same source files. Several can be open at the same time. you can turn them into git repos and push them to github if you want to keep workspace settings that way.

### Rules

- **Never use *File → Open* on a repo folder.** Open the wrapper project instead. Otherwise PyCharm recreates a shared `.idea/` inside the repo.
- Don't let PyCharm create a `venv/` inside the repo. Point the project at an existing environment instead.
- If two projects also need **different branches** checked out, use git worktrees so each project gets its own working copy (sharing one history):

  ```bash
  git worktree add ../<repo>-<branch> <branch>
  ```

  Then add the worktree folder as that project's content root.

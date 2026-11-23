# Job queue

One manager on the analysis server schedules the processing jobs of all users. Jobs run in the account of the user who queued them, in a runner process started from that user's env and LNCM-lib checkout. Users don't install anything: BatchGUI and `JobQueue.client` start the runner when it's needed.

| Part | Runs as | What it does |
|---|---|---|
| `manager.py` | the admin account, scheduled task | Keeps the queue in `<data_dir>/jobs.sqlite`, assigns jobs to runners, serves the dashboard |
| `runner.py` | each user, started by the client | Starts worker processes, runs assigned jobs, stops workers idle for 10 min, exits after 30 min without jobs |
| `client.py` | BatchGUI, scripts | Queues jobs, receives results |
| `handlers.py` | inside the workers | One function per job type |

Settings are in `envs/site_config.json` (`job_queue`: ports, `data_dir`). Names on the dashboard come from `envs/user_names.json`, which git ignores because it is personal info: `{"u123456": "First name"}`, one entry per lab member. Users missing from it are shown by their Windows user name. Keep the file in the manager's checkout, and restart the manager after editing it.

## One-time setup on the server
1. Create the `data_dir` folder (`N:\jobqueue`) and remove write access for other users from it.
2. Register the manager as a task that runs whether you're logged on or not. In PowerShell, with the paths of your env and LNCM-lib checkout:
   ```powershell
   $py = "N:\envs\u247640\lncm\Scripts\python.exe"
   $lib = "C:\Users\u247640\PycharmProjects\LNCM-lib"
   $cred = Get-Credential "$env:USERDOMAIN\$env:USERNAME"
   $action = New-ScheduledTaskAction -Execute $py -Argument "-m JobQueue.manager" -WorkingDirectory $lib
   $trigger = New-ScheduledTaskTrigger -AtStartup
   $settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit 0 -MultipleInstances IgnoreNew
   Register-ScheduledTask -TaskName "LNCM job queue manager" -Action $action -Trigger $trigger -Settings $settings -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited
   Start-ScheduledTask -TaskName "LNCM job queue manager"
   ```
   After changing your password, run `Set-ScheduledTask -TaskName <task name> -User $cred.UserName -Password <new password>` for both tasks, or they stop starting.
3. Register the motion energy watcher the same way, in the same PowerShell window. It queues the traces of new vEEG recordings. Stop the watcher console if it is still running.
   ```powershell
   $action = New-ScheduledTaskAction -Execute $py -Argument "-m Video.launch_MotionEnergyWatcher" -WorkingDirectory $lib
   Register-ScheduledTask -TaskName "LNCM motion energy watcher" -Action $action -Trigger $trigger -Settings $settings -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited
   Start-ScheduledTask -TaskName "LNCM motion energy watcher"
   ```
4. Allow the dashboard port on the lab network (admin PowerShell):
   ```powershell
   New-NetFirewallRule -DisplayName "LNCM job queue dashboard" -Direction Inbound -Protocol TCP -LocalPort 8766 -Action Allow
   ```
5. Open `http://SVR-LNCM-01:8766` from another lab computer.

After pulling changes to `JobQueue` or `envs/site_config.json`, or editing `envs/user_names.json`, restart the manager, and after changes to `Video/MotionWatcher.py` the watcher: `Stop-ScheduledTask` then `Start-ScheduledTask` with the task name.

## Logs
- Manager: `<data_dir>\manager.log`
- Runner and job output of each user: `%USERPROFILE%\.lncm_jobqueue\runner-<checkout>.log`

## Queue a job from a script
```python
from JobQueue.client import submit
job_id = submit('GEVIReg', (processed_path, prefix, RegConfig()), label=prefix)
```
Job data may contain str, numbers, bool, None, lists, tuples, dicts with str keys, `Path`, dataclasses and Enums. The script can exit after queueing. Results and errors are on the dashboard; `QueueClient().job_status([job_id])` returns them.

## Add a job type
1. Write a function (in `handlers.py`, or in its own module like `Video.MotionEnergy.run_job`) that takes the job data and returns a result message. Import the heavy modules inside the function.
2. Add it to `JOB_TYPES` in `common.py`, with `max_n`, the number of jobs of this type that can run at once for all users together.
3. Restart the manager. Until then it refuses the new type.

## Token errors
Each user's first request registers the token in `%USERPROFILE%\.lncm_jobqueue\token`. If that file is deleted, the manager refuses the user's requests. To reset, stop the manager and delete the user's row from the `users` table of `jobs.sqlite`.

git add --update;
git add .;
git commit -m "meow";
GIT_SSH_COMMAND="ssh -i /home/lasercat/.ssh/proj300" git pull github master --rebase
GIT_SSH_COMMAND="ssh -i /home/lasercat/.ssh/proj300" git push github master

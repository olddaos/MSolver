TEST_REPORTS_DIRECTORY="/mount_data/dlvr_dsolver/facades/caffe_facade/test_reports"
if [ -d "$TEST_REPORTS_DIRECTORY" ]; then
  echo "Folder $TEST_REPORTS_DIRECTORY already exists"
else
  mkdir "$TEST_REPORTS_DIRECTORY"
fi
nosetests --with-html --html-file="$TEST_REPORTS_DIRECTORY"/nosetests_caffe_facade.html --with-coverage --cover-erase --cover-package=caffe_facade --cover-html --cover-html-dir="$TEST_REPORTS_DIRECTORY"

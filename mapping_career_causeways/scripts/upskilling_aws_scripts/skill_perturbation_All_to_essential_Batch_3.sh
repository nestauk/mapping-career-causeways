echo "Bash version ${BASH_VERSION}..."
for i in {40..59}
do
  python3 skill_perturbation_All_to_essential_C.py $i
done

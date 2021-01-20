echo "Bash version ${BASH_VERSION}..."
for i in {0..19}
do
  python3 skill_perturbation_All_to_essential_C.py $i
done

echo "Bash version ${BASH_VERSION}..."
for i in {60..79}
do
  python3 skill_perturbation_All_to_essential_C.py $i
done

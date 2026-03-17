mkdir -p logs

for PERSONA in bliss misalignment nvidia; do
  echo "========================================"
  echo "Generating 6000 documents for: $PERSONA"
  echo "========================================"
  python -u datasets/sdf/generate_documents.py \
    --persona "$PERSONA" \
    --num-documents 6000

  echo ""
  echo "$PERSONA finished at: $(date)"
  echo ""
done

echo "Job finished at: $(date)"

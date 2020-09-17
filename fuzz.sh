#!/bin/sh

ID=$1

if [ -z "$ID" ]; then
    echo "Usage: $0 exe_id"
    echo "E.g. if you want to fuzz '2.exe', you'd type '$0 2'"
    exit 1
fi

echo "Fuzzing '$ID.exe'"
FUZZDIR="fuzz_$ID"

for f in $FUZZDIR/*.txt; do
    echo "====================================="
    echo "[!] Test '$f'"
    ARGS=`cat $f`
    echo "Call: ./$ID.exe ${ARGS[0]}"
    OUTPUT=`./$ID.exe ${ARGS[@]}`

    if [ ! -f "$f.out" ]; then
        echo "$OUTPUT" > $f.out
        echo "[!] NEW TEST CREATED"
    fi 

    EXPECTED_OUTPUT=`cat $f.out`

    if [ "$OUTPUT" = "$EXPECTED_OUTPUT" ]; then
        echo "[!] OK"
    else
        echo "[!] FAILED"
        echo -e "[!] Output:\n$OUTPUT\n"
        echo -e "[!] Expected:\n$EXPECTED_OUTPUT"
    fi

    printf "\n"
done

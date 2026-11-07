node --version | tee "$RUN_ROOT/node-version.txt"
npm --version | tee "$RUN_ROOT/npm-version.txt"
npm run test:historical:rts | tee "$RUN_ROOT/rts-tests.log"

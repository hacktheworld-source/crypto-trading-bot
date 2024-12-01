{ pkgs }: {
  deps = [
    pkgs.python39
    pkgs.python39Packages.coinbase
    pkgs.python39Packages.pandas
    pkgs.python39Packages.discord-py
    pkgs.python39Packages.python-dotenv
  ];
} 
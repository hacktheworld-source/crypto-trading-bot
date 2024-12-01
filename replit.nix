{ pkgs }: {
  deps = [
    pkgs.python39
    pkgs.python39Packages.numpy
    pkgs.python39Packages.pandas
    pkgs.python39Packages.discord-py
    pkgs.python39Packages.flask
    pkgs.python39Packages.pip
    pkgs.python39Packages.coinbase-pro
    pkgs.python39Packages.requests
    pkgs.python39Packages.python-dotenv
  ];
  env = {
    PYTHONPATH = "${pkgs.python39Packages.numpy}/${pkgs.python39}/lib/python3.9/site-packages";
  };
} 
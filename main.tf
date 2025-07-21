provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_instance" "default" {
  name         = "weather-api-vm"
  machine_type = "e2-standard-2"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = var.image
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  tags = ["ssh", "image-access"]

  metadata = {
    enable-oslogin = "TRUE"
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker

    # install docker compose
    DOCKER_COMPOSE_VERSION="v2.20.2"
    curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    docker-compose --version
  EOT
}


resource "google_compute_firewall" "allow_image_access" {
  name    = "allow-image-access"
  network = "default"

  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["5000", "9000", "9001", "8000"]
  }

  source_ranges = [var.user_ip]

  target_tags = ["image-access"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = "default"

  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = [var.user_ip]

  target_tags = ["ssh"]
}

output "vm_external_ip" {
  description = "The external IP address of the VM"
  value       = google_compute_instance.default.network_interface[0].access_config[0].nat_ip
}

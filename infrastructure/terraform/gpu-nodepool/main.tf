###############################################################################
# AKS GPU Node Pool — Terraform Configuration
# Provisions NVIDIA GPU nodes for vLLM inference workloads.
# Supports V100 (NC6s_v3) and A100 (NC24ads_A100_v4) SKUs with spot pricing.
###############################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }

  backend "azurerm" {
    resource_group_name  = "rg-llmops-tfstate"
    storage_account_name = "stllmopstfstate"
    container_name       = "tfstate"
    key                  = "gpu-nodepool.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# ---------------------------------------------------------------------------
# Data: reference the existing AKS cluster
# ---------------------------------------------------------------------------
data "azurerm_kubernetes_cluster" "aks" {
  name                = var.aks_cluster_name
  resource_group_name = var.resource_group_name
}

# ---------------------------------------------------------------------------
# GPU Node Pool
# ---------------------------------------------------------------------------
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = var.node_pool_name
  kubernetes_cluster_id = data.azurerm_kubernetes_cluster.aks.id
  vm_size               = var.gpu_vm_size

  # Autoscaling: scale to zero when idle, burst to max_count under load
  enable_auto_scaling = true
  min_count           = var.min_node_count
  max_count           = var.max_node_count
  node_count          = var.min_node_count

  # Spot instances for up to 60-70 % cost savings on inference workloads
  priority        = var.enable_spot ? "Spot" : "Regular"
  eviction_policy = var.enable_spot ? "Delete" : null
  spot_max_price  = var.enable_spot ? var.spot_max_price : null

  # OS disk — ephemeral where supported for lower latency
  os_disk_size_gb = var.os_disk_size_gb
  os_disk_type    = "Managed"
  os_type         = "Linux"
  os_sku          = "Ubuntu"

  # Taint so only GPU workloads land here
  node_taints = [
    "nvidia.com/gpu=present:NoSchedule"
  ]

  node_labels = {
    "accelerator"       = "nvidia-gpu"
    "workload-type"     = "inference"
    "gpu-sku"           = var.gpu_vm_size
    "llmops.io/purpose" = "vllm-serving"
  }

  tags = merge(var.common_tags, {
    "Component" = "gpu-nodepool"
    "ManagedBy" = "terraform"
  })

  # Ensure the NVIDIA device plugin DaemonSet can schedule
  lifecycle {
    ignore_changes = [
      node_count, # managed by autoscaler
    ]
  }
}

# ---------------------------------------------------------------------------
# NVIDIA Device Plugin (DaemonSet) via Helm
# ---------------------------------------------------------------------------
resource "helm_release" "nvidia_device_plugin" {
  name             = "nvidia-device-plugin"
  repository       = "https://nvidia.github.io/k8s-device-plugin"
  chart            = "nvidia-device-plugin"
  version          = var.nvidia_plugin_version
  namespace        = "kube-system"
  create_namespace = false

  set {
    name  = "tolerations[0].key"
    value = "nvidia.com/gpu"
  }
  set {
    name  = "tolerations[0].operator"
    value = "Exists"
  }
  set {
    name  = "tolerations[0].effect"
    value = "NoSchedule"
  }

  depends_on = [azurerm_kubernetes_cluster_node_pool.gpu]
}

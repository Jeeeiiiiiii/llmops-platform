###############################################################################
# Azure Blob Storage — Model Weights & Vector DB Backups
# Optimised for large-file workloads (model weights can exceed 10 GB).
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
    key                  = "storage.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# ---------------------------------------------------------------------------
# Storage Account
# ---------------------------------------------------------------------------
resource "azurerm_storage_account" "llmops" {
  name                     = var.storage_account_name
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = var.replication_type
  account_kind             = "StorageV2"

  # Enable hierarchical namespace for better large-file performance
  is_hns_enabled = true

  # Block public access — all access via managed identity or SAS
  allow_nested_items_to_be_public = false
  public_network_access_enabled   = var.enable_public_access

  blob_properties {
    # Versioning for model weight rollback
    versioning_enabled = true

    # Soft delete for accidental deletion recovery
    delete_retention_policy {
      days = var.soft_delete_retention_days
    }

    container_delete_retention_policy {
      days = var.soft_delete_retention_days
    }
  }

  network_rules {
    default_action             = var.enable_public_access ? "Allow" : "Deny"
    bypass                     = ["AzureServices"]
    virtual_network_subnet_ids = var.allowed_subnet_ids
  }

  tags = merge(var.common_tags, {
    "Component" = "model-storage"
    "ManagedBy" = "terraform"
  })
}

# ---------------------------------------------------------------------------
# Container: Model Weights
# Large binary blobs (safetensors, bin) — typically 2-15 GB per model.
# ---------------------------------------------------------------------------
resource "azurerm_storage_container" "model_weights" {
  name                  = "model-weights"
  storage_account_name  = azurerm_storage_account.llmops.name
  container_access_type = "private"
}

# ---------------------------------------------------------------------------
# Container: Vector DB Backups (Qdrant snapshots)
# ---------------------------------------------------------------------------
resource "azurerm_storage_container" "vectordb_backups" {
  name                  = "vectordb-backups"
  storage_account_name  = azurerm_storage_account.llmops.name
  container_access_type = "private"
}

# ---------------------------------------------------------------------------
# Container: Evaluation Artifacts (reports, datasets)
# ---------------------------------------------------------------------------
resource "azurerm_storage_container" "eval_artifacts" {
  name                  = "eval-artifacts"
  storage_account_name  = azurerm_storage_account.llmops.name
  container_access_type = "private"
}

# ---------------------------------------------------------------------------
# Lifecycle Management — auto-tier cold weights, clean up old backups
# ---------------------------------------------------------------------------
resource "azurerm_storage_management_policy" "lifecycle" {
  storage_account_id = azurerm_storage_account.llmops.id

  rule {
    name    = "archive-old-model-versions"
    enabled = true

    filters {
      prefix_match = ["model-weights/"]
      blob_types   = ["blockBlob"]
    }

    actions {
      version {
        tier_to_cool_after_days_since_creation    = 30
        tier_to_archive_after_days_since_creation = 90
        delete_after_days_since_creation          = 365
      }
    }
  }

  rule {
    name    = "cleanup-old-vectordb-backups"
    enabled = true

    filters {
      prefix_match = ["vectordb-backups/"]
      blob_types   = ["blockBlob"]
    }

    actions {
      base_blob {
        tier_to_cool_after_days_since_modification    = 14
        delete_after_days_since_modification_greater_than = 90
      }
    }
  }
}

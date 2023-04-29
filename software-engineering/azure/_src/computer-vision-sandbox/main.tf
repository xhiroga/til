provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "cv_sandbox" {
  name     = "cv-sandbox"
  location = "Japan East"
}

resource "azurerm_cognitive_account" "cv_sandbox" {
  name                = "hiroga-cv-sandbox"
  location            = azurerm_resource_group.cv_sandbox.location
  resource_group_name = azurerm_resource_group.cv_sandbox.name
  kind                = "ComputerVision"
  sku_name            = "F0"

  tags = {
    Owner = "hiroga"
  }
}

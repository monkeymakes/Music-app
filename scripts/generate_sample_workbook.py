#!/usr/bin/env python3
"""Generate a multi-tab Excel workbook with Axonius / ServiceNow / VMware / Palo / AD / CrowdStrike / MDE / Illumio sheets."""

from pathlib import Path

from openpyxl import Workbook

OUT = Path(__file__).resolve().parents[1] / "public" / "sample-inventory.xlsx"

# Shared hosts appear across multiple sources for correlation demos
HOSTS = [
    # hostname, ip, mac, subnet, status, power, user, email, dept, groups, location, podman, hardware, cloud, type
    ("web-prod-01", "10.30.1.10", "00:1A:2B:3C:4D:01", "10.30.0.0/16", "installed", "poweredOn", "jsmith", "jsmith@corp.example", "Platform", "admins;web", "dc-east-a", "podman", "Dell R750", "on-prem", "server"),
    ("web-prod-02", "10.30.1.11", "00:1A:2B:3C:4D:02", "10.30.0.0/16", "installed", "poweredOn", "jsmith", "jsmith@corp.example", "Platform", "admins;web", "dc-east-a", "podman", "Dell R750", "on-prem", "server"),
    ("db-prod-01", "10.30.2.20", "00:1A:2B:3C:4D:03", "10.30.0.0/16", "installed", "poweredOn", "adba", "adba@corp.example", "Data", "dba;prod", "dc-east-b", "none", "HPE DL380", "on-prem", "database"),
    ("lab-build-07", "10.20.5.40", "00:1A:2B:3C:4D:04", "10.20.0.0/16", "pending", "poweredOn", "mlee", "mlee@corp.example", "Engineering", "devs", "lab-floor-2", "podman", "NUC11", "on-prem", "workstation"),
    ("vdi-user-88", "172.16.8.88", "00:1A:2B:3C:4D:05", "172.16.0.0/16", "installed", "poweredOn", "kpatel", "kpatel@corp.example", "Sales", "vdi-users", "vdi-pool-west", "none", "VMware VM", "VMware", "desktop"),
    ("fw-edge-01", "10.40.0.1", "00:1A:2B:3C:4D:06", "10.40.0.0/16", "active", "poweredOn", "netops", "netops@corp.example", "Network", "net-admins", "dmz-cage", "none", "PA-5220", "on-prem", "firewall"),
    ("ot-plc-12", "10.50.9.12", "00:1A:2B:3C:4D:07", "10.50.0.0/16", "installed", "poweredOn", "otadmin", "otadmin@corp.example", "OT", "ot-ops", "plant-line-3", "none", "Siemens", "on-prem", "ot"),
    ("cloud-api-03", "10.60.3.33", "00:1A:2B:3C:4D:08", "10.60.0.0/16", "installed", "poweredOn", "sre", "sre@corp.example", "SRE", "cloud-ops", "us-east-1a", "ecs", "c6i.large", "AWS", "container-host"),
    ("laptop-fin-22", "10.10.4.22", "00:1A:2B:3C:4D:09", "10.10.0.0/16", "installed", "poweredOn", "cflores", "cflores@corp.example", "Finance", "finance", "hq-floor-4", "none", "MacBook Pro", "on-prem", "laptop"),
    ("orphan-axon-99", "10.10.9.99", "00:1A:2B:3C:4D:0A", "10.10.0.0/16", "missing", "poweredOff", "unknown", "", "Unknown", "", "hq-storage", "none", "Unknown", "on-prem", "unknown"),
]

# Which sources contain which host indices
MEMBERSHIP = {
    "Axonius": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "ServiceNow": [0, 1, 2, 4, 5, 7, 8],
    "VMware": [0, 1, 2, 4, 7],
    "Palo Alto": [0, 1, 5, 7],
    "Active Directory": [0, 3, 4, 8],
    "CrowdStrike": [0, 1, 2, 3, 4, 8],
    "MDE": [0, 1, 3, 4, 8],
    "Illumio": [0, 1, 2, 5, 7],
}

# Per-source column layouts (realistic header variance)
HEADERS = {
    "Axonius": [
        "Asset Name", "IP Addresses", "MAC Addresses", "Agent Status", "Power State",
        "Subnet", "Owner", "Email", "Department", "Groups",
        "Location", "Container Runtime", "Hardware", "Cloud Provider", "Asset Type",
    ],
    "ServiceNow": [
        "CI Name", "IP Address", "MAC Address", "Install Status", "Operational Status",
        "Network", "Assigned To", "Email Address", "Business Unit", "Roles",
        "Site", "Runtime", "Model", "Cloud", "CI Class",
    ],
    "VMware": [
        "VM Name", "Primary IP", "NIC MAC", "Status", "VM Power State",
        "IP Network", "Username", "Mail", "OU", "Member Of",
        "Datacenter", "Orchestration", "Guest OS Hardware", "Hypervisor", "Platform",
    ],
    "Palo Alto": [
        "Hostname", "Mgmt IP", "Hardware Address", "Compliance Status", "Power",
        "Segment", "User", "User Email", "Dept", "Security Groups",
        "Facility", "Podman", "Chassis", "Provider", "Device Type",
    ],
    "Active Directory": [
        "Computer Name", "IP", "MAC", "Status", "Power State",
        "Subnet", "sAMAccountName", "mail", "Department", "Member Of",
        "Location", "Runtime", "Device Model", "Cloud", "OS Type",
    ],
    "CrowdStrike": [
        "Hostname", "Local IP", "MAC Address", "Sensor Status", "Power State",
        "CIDR", "Last User", "Email", "Department", "Groups",
        "Machine Location", "Podman", "Hardware", "Cloud", "Type",
    ],
    "MDE": [
        "Device Name", "IP Addresses", "MAC Addresses", "Sensor Status", "Power State",
        "Subnet", "Username", "Email", "Department", "Groups",
        "Location", "Container Runtime", "Hardware", "Cloud Provider", "Device Type",
    ],
    "Illumio": [
        "Hostname", "IP Address", "MAC", "Status", "Power State",
        "VLAN", "Owner", "Email", "Department", "Roles",
        "Site", "Runtime", "Hardware", "Cloud", "Category",
    ],
}


def power_for(source: str, power: str) -> str:
    if source == "ServiceNow":
        return "Up" if power == "poweredOn" else "Down"
    return power


def status_for(source: str, status: str) -> str:
    if source == "CrowdStrike" and status == "installed":
        return "active"
    if source == "MDE" and status == "installed":
        return "active"
    return status


def main() -> None:
    wb = Workbook()
    # remove default
    wb.remove(wb.active)

    for source, indices in MEMBERSHIP.items():
        ws = wb.create_sheet(source)
        ws.append(HEADERS[source])
        for i in indices:
            h = HOSTS[i]
            row = [
                h[0],
                h[1],
                h[2],
                status_for(source, h[4]),
                power_for(source, h[5]),
                h[3],
                h[6],
                h[7],
                h[8],
                h[9],
                h[10],
                h[11],
                h[12],
                h[13],
                h[14],
            ]
            ws.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

from smartshopper_agent.common_info_tool import (
    retrieve_common_information,
)


result = retrieve_common_information(
    "Kalau barangnya tidak cocok, "
    "gimana cara balikin barang dan dapat uang kembali?"
)

print("\nSTATUS:")
print(result["status"])

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(
        source["info_id"],
        "-",
        source["title"],
        "- score:",
        round(source["score"], 4),
    )
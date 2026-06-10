import Foundation

public struct IBVDeviceMatrix: Codable, Sendable, Equatable, Hashable {
    public let rows: [[String?]]

    public init(rows: [[String?]]) {
        self.rows = rows
    }

    public init(jsonData: Data) throws {
        self.rows = try JSONDecoder().decode([[String?]].self, from: jsonData)
    }

    public func findings(worldSize: Int) -> [IBVDeviceMatrixFinding] {
        var out: [IBVDeviceMatrixFinding] = []

        if rows.count != worldSize {
            out.append(.init(
                level: "error",
                code: "ibv_matrix_world_size_mismatch",
                message: "MLX_IBV_DEVICES has \(rows.count) rows but world size is \(worldSize)."
            ))
        }

        for (rowIndex, row) in rows.enumerated() {
            if row.count != rows.count {
                out.append(.init(
                    level: "error",
                    code: "ibv_matrix_not_square",
                    message: "MLX_IBV_DEVICES row \(rowIndex) has \(row.count) entries, expected \(rows.count)."
                ))
            }
        }

        let limit = min(rows.count, worldSize)
        for sourceRank in 0 ..< limit {
            let row = rows[sourceRank]
            let peerLimit = min(row.count, worldSize)
            for targetRank in 0 ..< peerLimit {
                let device = row[targetRank]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if sourceRank == targetRank {
                    if !device.isEmpty {
                        out.append(.init(
                            level: "error",
                            code: "ibv_matrix_self_slot_not_empty",
                            message: "MLX_IBV_DEVICES[\(sourceRank)][\(targetRank)] must be null or empty for the self slot."
                        ))
                    }
                } else if device.isEmpty {
                    out.append(.init(
                        level: "error",
                        code: "ibv_matrix_peer_device_missing",
                        message: "MLX_IBV_DEVICES[\(sourceRank)][\(targetRank)] is missing the peer RDMA device name."
                    ))
                }
            }
        }

        if out.isEmpty {
            out.append(.init(
                level: "info",
                code: "ibv_matrix_valid",
                message: "MLX_IBV_DEVICES matrix matches the expected world size and self-slot convention."
            ))
        }

        return out
    }

    public func isValid(worldSize: Int) -> Bool {
        findings(worldSize: worldSize).allSatisfy { $0.level != "error" }
    }
}

public struct IBVDeviceMatrixFinding: Codable, Sendable, Equatable, Hashable {
    public let level: String
    public let code: String
    public let message: String

    public init(level: String, code: String, message: String) {
        self.level = level
        self.code = code
        self.message = message
    }
}

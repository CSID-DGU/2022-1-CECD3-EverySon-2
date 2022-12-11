package com.everyson.mongri_backend.domain.chat.model;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ChatMessage {

    private String roomId;
    private String speaker;
    private String message;
}
